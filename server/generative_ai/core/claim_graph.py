"""
claim_graph.py

LangGraph-based claim generation pipeline. Replaces the sequential
generate_raw() calls in app.py with a proper stateful graph.

Nodes:
  1. segment_damage     - Run CV models on uploaded images
  2. analyze_damage      - Convert masks to structured damage text + repair estimates
  3. resolve_policy      - Look up insurance coverage from the policy engine
  4. generate_claim_letter   - LLM: write the claim letter
  5. generate_damage_report  - LLM: write the damage analysis
  6. generate_coverage       - LLM: assess policy coverage
  7. generate_action_plan    - LLM: write personalized next steps

State flows through all nodes via a shared ClaimState TypedDict.
Tracing is handled via LangFuse or LangSmith callbacks -- just set
the environment variables and pass the config.

Usage:
    from generative_ai.core.claim_graph import build_claim_graph, ClaimState

    graph = build_claim_graph(llm_client)
    result = graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})
"""

import logging
import time
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import numpy as np
from PIL import Image

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger("claim_graph")


# ─── State Schema ────────────────────────────────────────────────

class ClaimState(TypedDict, total=False):
    """
    Shared state that flows through the entire claim pipeline.
    Each node reads what it needs and writes its outputs.
    """
    # ── Inputs (set before graph invocation) ──
    images: List[Any]                  # PIL Images
    seg_model_name: str                # Segmentation model checkpoint name
    user_info: Dict[str, str]          # user_name, user_address, user_phone, user_email
    vehicle_info: Dict[str, str]       # vehicle_year, vehicle_make, vehicle_model, vehicle_vin, license_plate
    incident_info: Dict[str, str]      # incident_date, incident_time, incident_location, event_description
    insurance_info: Dict[str, str]     # insurance_company, policy_number
    provider_id: str                   # Internal provider ID (e.g., 'state_farm')
    plan_id: str                       # Internal plan ID (e.g., 'standard')

    # ── CV Pipeline Outputs ──
    masks: List[Any]                   # Binary numpy masks from segmentation
    overlayed_images: List[Any]        # Numpy arrays with damage overlays
    overlayed_pil: List[Any]           # PIL versions of overlays (for VL model)
    damage_text: str                   # Structured damage description
    damage_analyses: List[Dict]        # Raw analysis dicts from damage_analyzer

    # ── Policy Engine Outputs ──
    policy_context: str                # Full policy text for LLM
    policy_summary: str                # Human-readable coverage summary
    deductible: int                    # Collision deductible amount
    provider_contact: Dict[str, str]   # claims_phone, claims_url

    # ── Repair Estimate Outputs ──
    repair_text: str                   # Formatted repair estimate
    repair_result: Dict[str, Any]      # Raw estimate data (totals, per-region)

    # ── LLM Outputs ──
    section_claim_letter: str
    section_damage_report: str
    section_coverage: str
    section_action_plan: str

    # ── Metadata ──
    timings: Dict[str, float]          # Per-node execution times
    errors: List[str]                  # Any errors encountered


# ─── Node Functions ──────────────────────────────────────────────

def node_segment_damage(state: ClaimState) -> dict:
    """Node 1: Run CV segmentation on all uploaded images."""
    from services.segmentation import segment_damage, overlay_mask

    t0 = time.time()
    images = state.get("images", [])
    model_name = state.get("seg_model_name", "")

    masks = []
    overlayed_images = []
    for img in images:
        img_array = np.array(img)
        mask = segment_damage(img_array, model_name=model_name)
        overlayed = overlay_mask(img, mask, color=(255, 50, 50), alpha=0.6)
        masks.append(mask)
        overlayed_images.append(overlayed)

    overlayed_pil = [Image.fromarray(ov) for ov in overlayed_images]

    elapsed = time.time() - t0
    logger.info(f"[segment_damage] Processed {len(images)} images in {elapsed:.1f}s")

    return {
        "masks": masks,
        "overlayed_images": overlayed_images,
        "overlayed_pil": overlayed_pil,
        "timings": {**state.get("timings", {}), "segment_damage": elapsed},
    }


def node_analyze_damage(state: ClaimState) -> dict:
    """Node 2: Convert masks to structured text + repair estimates."""
    from services.damage_analyzer import generate_detected_damage, analyze_masks
    from services.repair_estimator import get_estimate_summary_for_llm

    t0 = time.time()
    masks = state.get("masks", [])
    deductible = state.get("deductible", 500)

    damage_analyses = analyze_masks(masks)
    damage_text = generate_detected_damage(masks)
    repair_text, repair_result = get_estimate_summary_for_llm(damage_analyses, deductible=deductible)

    elapsed = time.time() - t0
    logger.info(f"[analyze_damage] Found {repair_result['num_regions']} damage regions in {elapsed:.1f}s")

    return {
        "damage_text": damage_text,
        "damage_analyses": damage_analyses,
        "repair_text": repair_text,
        "repair_result": repair_result,
        "timings": {**state.get("timings", {}), "analyze_damage": elapsed},
    }


def node_resolve_policy(state: ClaimState) -> dict:
    """Node 3: Look up insurance coverage from the policy engine."""
    from services.policy_engine import (
        resolve_policy, generate_policy_context,
        format_coverage_summary, get_provider_contact,
    )

    t0 = time.time()
    provider_id = state.get("provider_id", "custom")
    plan_id = state.get("plan_id", "default")

    resolved = resolve_policy(provider_id, plan_id)
    policy_context = generate_policy_context(provider_id, plan_id)
    policy_summary = format_coverage_summary(resolved) if resolved else ""
    deductible = 500
    if resolved:
        deductible = resolved.get("coverage", {}).get("collision", {}).get("deductible", 500)
    provider_contact = get_provider_contact(provider_id)

    elapsed = time.time() - t0
    logger.info(f"[resolve_policy] Resolved {provider_id}/{plan_id} in {elapsed:.1f}s")

    return {
        "policy_context": policy_context,
        "policy_summary": policy_summary,
        "deductible": deductible,
        "provider_contact": provider_contact,
        "timings": {**state.get("timings", {}), "resolve_policy": elapsed},
    }


def _make_llm_node(node_name: str, output_key: str, prompt_builder, llm_client, uses_images: bool = False, max_tokens: int = 1000):
    """
    Factory function that creates an LLM node.
    The prompt_builder is a callable(state) -> str that builds the prompt.
    The llm_client is captured via closure -- not passed through state.
    """
    def llm_node(state: ClaimState) -> dict:
        t0 = time.time()
        if llm_client is None:
            return {output_key: "[ERROR: LLM client not available]", "errors": state.get("errors", []) + [f"{node_name}: no LLM client"]}

        prompt = prompt_builder(state)
        images = state.get("overlayed_pil", []) if uses_images else None

        try:
            result = llm_client.generate_raw(prompt, images=images, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"[{node_name}] LLM error: {e}")
            result = f"[ERROR generating {node_name}: {e}]"

        elapsed = time.time() - t0
        logger.info(f"[{node_name}] Generated in {elapsed:.1f}s")

        return {
            output_key: result,
            "timings": {**state.get("timings", {}), node_name: elapsed},
        }

    llm_node.__name__ = node_name
    return llm_node


# ─── Prompt Builders ─────────────────────────────────────────────

def _prompt_claim_letter(state: ClaimState) -> str:
    u = state.get("user_info", {})
    v = state.get("vehicle_info", {})
    i = state.get("incident_info", {})
    ins = state.get("insurance_info", {})
    damage_text = state.get("damage_text", "")

    return f"""Write a formal insurance claim letter written by {u.get('user_name','')} to {ins.get('insurance_company','')}.

RULES: First person as {u.get('user_name','')}. No technical data (bounding boxes, percentages). No invented facts. You are the claimant, not the insurer.

CLAIMANT: {u.get('user_name','')}, {u.get('user_address','')}, Phone: {u.get('user_phone','')}, Email: {u.get('user_email','')}
INSURANCE: {ins.get('insurance_company','')}, Policy: {ins.get('policy_number','')}
VEHICLE: {v.get('vehicle_year','')} {v.get('vehicle_make','')} {v.get('vehicle_model','')}, VIN: {v.get('vehicle_vin','')}, Plate: {v.get('license_plate','')}
INCIDENT: {i.get('incident_date','')} at {i.get('incident_time','')}, {i.get('incident_location','')}
WHAT HAPPENED: {i.get('event_description','')}
DAMAGE (AI): {damage_text}

FORMAT:
{u.get('user_name','')}
{u.get('user_address','')}
Phone: {u.get('user_phone','')} | Email: {u.get('user_email','')}
Date: {i.get('incident_date','')}

{ins.get('insurance_company','')} - Claims Department
Policy: {ins.get('policy_number','')}
Subject: Insurance Claim for Vehicle Damage

Dear Claims Department,
[File a claim. Describe incident. Summarize damage in plain English. Mention attached photos. Request claim be opened.]

Sincerely,
{u.get('user_name','')}
{u.get('user_phone','')} | {u.get('user_email','')}"""


def _prompt_damage_report(state: ClaimState) -> str:
    damage_text = state.get("damage_text", "")
    repair_text = state.get("repair_text", "")
    repair_result = state.get("repair_result", {})
    num_regions = repair_result.get("num_regions", 0)

    return f"""Automotive damage assessor. Attached images show damage in red.

AI DATA: {damage_text}
REPAIR ESTIMATE: {repair_text}

Write a report for ALL {num_regions} damage regions. For each:
1. Affected area (automotive terms)
2. Damage type: dent/scratch/crack/deformation/shatter/structural
3. Severity with explanation
4. Estimated repair cost range
5. Claim relevance

Cover ALL regions. Do NOT stop after one."""


def _prompt_coverage(state: ClaimState) -> str:
    policy_context = state.get("policy_context", "")
    damage_text = state.get("damage_text", "")
    repair_result = state.get("repair_result", {})

    return f"""Insurance policy analyst. Read the policy CAREFULLY.

POLICY: {policy_context}
DAMAGES: {damage_text}
REPAIR ESTIMATE: ${repair_result.get('total_low', 0):,} - ${repair_result.get('total_high', 0):,}

Answer using ONLY the policy:
1. COVERED ITEMS: Which damages are covered? Reference policy language.
2. EXCLUSIONS: Only those explicitly stated. If none apply, say so.
3. DEDUCTIBLE: Quote exact amount. Calculate insurer pays = estimate - deductible.
4. RENTAL COVERAGE: Included? Daily/duration limits?
5. CLAIM FILING: Deadlines and requirements?
6. CONFIDENCE: High/Medium/Low with justification.

Do NOT invent policy terms."""


def _prompt_action_plan(state: ClaimState) -> str:
    u = state.get("user_info", {})
    v = state.get("vehicle_info", {})
    i = state.get("incident_info", {})
    ins = state.get("insurance_info", {})
    repair_result = state.get("repair_result", {})
    deductible = state.get("deductible", 500)
    contact = state.get("provider_contact", {})

    return f"""5 action items for {u.get('user_name','')} after their accident.

Context: Accident {i.get('incident_date','')} at {i.get('incident_location','')}. Insurer: {ins.get('insurance_company','')} (#{ins.get('policy_number','')}). Claims phone: {contact.get('claims_phone','')}. Vehicle: {v.get('vehicle_year','')} {v.get('vehicle_make','')} {v.get('vehicle_model','')}. Repair estimate: ${repair_result.get('total_midpoint', 0):,}. Deductible: ${deductible}.

Numbered list only. No greeting/sign-off. "You" language. Mention {ins.get('insurance_company','')} by name."""


# ─── Graph Builder ───────────────────────────────────────────────

def build_claim_graph(llm_client=None):
    """
    Build and compile the claim generation LangGraph.

    Args:
        llm_client: QwenVLLLMClient instance (or any client with generate_raw method).
                    If None, LLM nodes will return error messages.

    Returns:
        Compiled StateGraph ready to invoke.
    """
    # Create LLM nodes using the factory -- client captured via closure
    node_claim_letter = _make_llm_node(
        "generate_claim_letter", "section_claim_letter",
        _prompt_claim_letter, llm_client, uses_images=True, max_tokens=1000,
    )
    node_damage_report = _make_llm_node(
        "generate_damage_report", "section_damage_report",
        _prompt_damage_report, llm_client, uses_images=True, max_tokens=1000,
    )
    node_coverage = _make_llm_node(
        "generate_coverage", "section_coverage",
        _prompt_coverage, llm_client, uses_images=False, max_tokens=700,
    )
    node_action_plan = _make_llm_node(
        "generate_action_plan", "section_action_plan",
        _prompt_action_plan, llm_client, uses_images=False, max_tokens=400,
    )

    # Build the graph
    builder = StateGraph(ClaimState)

    # Add all nodes
    builder.add_node("segment_damage", node_segment_damage)
    builder.add_node("analyze_damage", node_analyze_damage)
    builder.add_node("resolve_policy", node_resolve_policy)
    builder.add_node("generate_claim_letter", node_claim_letter)
    builder.add_node("generate_damage_report", node_damage_report)
    builder.add_node("generate_coverage", node_coverage)
    builder.add_node("generate_action_plan", node_action_plan)

    # Define edges
    # START → segment_damage → analyze_damage
    builder.add_edge(START, "segment_damage")
    builder.add_edge("segment_damage", "analyze_damage")

    # After analyze_damage, resolve_policy runs
    # (we need deductible from policy before repair estimate, but
    #  analyze_damage can use a default deductible -- policy resolution
    #  is fast so we run it after segmentation)
    builder.add_edge("analyze_damage", "resolve_policy")

    # After policy is resolved, run all 4 LLM stages sequentially
    # (sequential because we're on CPU with a single model)
    builder.add_edge("resolve_policy", "generate_claim_letter")
    builder.add_edge("generate_claim_letter", "generate_damage_report")
    builder.add_edge("generate_damage_report", "generate_coverage")
    builder.add_edge("generate_coverage", "generate_action_plan")
    builder.add_edge("generate_action_plan", END)

    # Compile
    graph = builder.compile()

    return graph


def invoke_claim_graph(
    graph,
    images: List,
    seg_model_name: str,
    user_info: Dict,
    vehicle_info: Dict,
    incident_info: Dict,
    insurance_info: Dict,
    provider_id: str,
    plan_id: str,
    callbacks: Optional[List] = None,
) -> ClaimState:
    """
    Convenience function to invoke the claim graph with properly
    structured initial state.

    Args:
        graph: Compiled claim graph from build_claim_graph().
        images: List of PIL Images.
        seg_model_name: Name of the segmentation model checkpoint.
        user_info: Dict with user_name, user_address, etc.
        vehicle_info: Dict with vehicle_year, etc.
        incident_info: Dict with incident_date, event_description, etc.
        insurance_info: Dict with insurance_company, policy_number.
        provider_id: Internal provider ID.
        plan_id: Internal plan ID.
        callbacks: Optional list of callback handlers (LangFuse, LangSmith).

    Returns:
        Final ClaimState with all sections generated.
    """
    initial_state: ClaimState = {
        "images": images,
        "seg_model_name": seg_model_name,
        "user_info": user_info,
        "vehicle_info": vehicle_info,
        "incident_info": incident_info,
        "insurance_info": insurance_info,
        "provider_id": provider_id,
        "plan_id": plan_id,
        "timings": {},
        "errors": [],
    }

    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    result = graph.invoke(initial_state, config=config)
    return result


def get_graph_image(graph) -> Optional[bytes]:
    """
    Generate a PNG visualization of the graph structure.
    Requires graphviz and pygraphviz to be installed.

    Returns:
        PNG bytes, or None if visualization is not available.
    """
    try:
        return graph.get_graph().draw_mermaid_png()
    except Exception:
        try:
            return graph.get_graph().draw_png()
        except Exception as e:
            logger.warning(f"Graph visualization not available: {e}")
            return None


def get_graph_mermaid(graph) -> str:
    """
    Generate a Mermaid diagram string for the graph.
    This can be rendered in Streamlit or any markdown viewer.
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Mermaid generation failed: {e}")
        return ""