"""
policy_engine.py

Loads insurance provider data and resolves policy details for a given
provider + plan combination. Generates structured policy context text
that the LLM uses for accurate coverage assessment.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_PROVIDERS_PATH = Path(__file__).parent / "insurance_providers.json"
_providers_cache: Optional[Dict] = None


def _load_providers() -> Dict:
    """Load and cache the insurance providers database."""
    global _providers_cache
    if _providers_cache is None:
        with open(_PROVIDERS_PATH, "r") as f:
            _providers_cache = json.load(f)
    return _providers_cache


def get_provider_list() -> List[Dict[str, str]]:
    """
    Return a list of available insurance providers for UI dropdowns.

    Returns:
        List of dicts with 'id' and 'display_name' keys.
    """
    db = _load_providers()
    return [
        {"id": pid, "display_name": pdata["display_name"]}
        for pid, pdata in db["providers"].items()
    ]


def get_plan_list(provider_id: str) -> List[Dict[str, str]]:
    """
    Return available plans for a given provider.

    Args:
        provider_id: Internal provider ID (e.g., 'state_farm').

    Returns:
        List of dicts with 'id' and 'plan_name' keys.
    """
    db = _load_providers()
    provider = db["providers"].get(provider_id)
    if not provider:
        return []
    return [
        {"id": plan_id, "plan_name": plan_data["plan_name"]}
        for plan_id, plan_data in provider["plans"].items()
    ]


def resolve_policy(provider_id: str, plan_id: str) -> Optional[Dict]:
    """
    Look up the full policy details for a provider + plan combination.

    Args:
        provider_id: Internal provider ID.
        plan_id: Internal plan ID (e.g., 'basic', 'standard', 'premium').

    Returns:
        Full policy dict including coverage, exclusions, conditions, etc.
        None if provider or plan not found.
    """
    db = _load_providers()
    provider = db["providers"].get(provider_id)
    if not provider:
        return None
    plan = provider["plans"].get(plan_id)
    if not plan:
        return None

    # Attach provider-level info to the resolved policy
    return {
        "provider_id": provider_id,
        "provider_name": provider["display_name"],
        "claims_phone": provider.get("claims_phone", ""),
        "claims_url": provider.get("claims_url", ""),
        "plan_id": plan_id,
        **plan,
    }


def get_provider_contact(provider_id: str) -> Dict[str, str]:
    """
    Return contact info for a provider (phone, URL).
    """
    db = _load_providers()
    provider = db["providers"].get(provider_id, {})
    return {
        "display_name": provider.get("display_name", "Unknown"),
        "claims_phone": provider.get("claims_phone", "[UNKNOWN]"),
        "claims_url": provider.get("claims_url", "[UNKNOWN]"),
    }


def format_coverage_summary(policy: Dict) -> str:
    """
    Convert a resolved policy into a human-readable coverage summary
    suitable for display in the UI.

    Args:
        policy: Resolved policy dict from resolve_policy().

    Returns:
        Formatted string summarizing key coverage details.
    """
    if not policy:
        return "No policy information available."

    coverage = policy.get("coverage", {})
    lines = []
    lines.append(f"Plan: {policy.get('plan_name', 'Unknown')}")
    lines.append(f"Provider: {policy.get('provider_name', 'Unknown')}")
    lines.append("")

    # Collision
    col = coverage.get("collision", {})
    if col.get("included"):
        lines.append(f"Collision Coverage: Included (Deductible: ${col.get('deductible', 'N/A')})")
        lines.append(f"  Limit: {col.get('limit', 'N/A')}")
    else:
        lines.append("Collision Coverage: Not included")

    # Comprehensive
    comp = coverage.get("comprehensive", {})
    if comp.get("included"):
        lines.append(f"Comprehensive Coverage: Included (Deductible: ${comp.get('deductible', 'N/A')})")
        lines.append(f"  Limit: {comp.get('limit', 'N/A')}")
    else:
        lines.append("Comprehensive Coverage: Not included")

    # Liability
    liab = coverage.get("liability", {})
    if liab:
        lines.append(
            f"Liability: ${liab.get('bodily_injury_per_person', 0):,}/"
            f"${liab.get('bodily_injury_per_accident', 0):,} BI, "
            f"${liab.get('property_damage', 0):,} PD"
        )

    # Rental
    rental = coverage.get("rental_reimbursement", {})
    if rental.get("included"):
        lines.append(
            f"Rental Reimbursement: Up to ${rental.get('daily_limit', 0)}/day "
            f"for {rental.get('max_days', 0)} days"
        )
    else:
        lines.append("Rental Reimbursement: Not included")

    # Roadside
    roadside = coverage.get("roadside_assistance", {})
    if roadside.get("included"):
        services = ", ".join(roadside.get("services", []))
        lines.append(f"Roadside Assistance: Included ({services})")
    else:
        lines.append("Roadside Assistance: Not included")

    # Glass
    glass = coverage.get("glass_coverage", {})
    if glass.get("included"):
        lines.append(f"Glass Coverage: Included (Deductible: ${glass.get('deductible', 'N/A')})")
    else:
        lines.append(f"Glass Coverage: {glass.get('description', 'Not included')}")

    return "\n".join(lines)


def generate_policy_context(provider_id: str, plan_id: str) -> str:
    """
    Generate the full policy context text that gets injected into the LLM
    prompt for coverage assessment. This replaces the old hardcoded
    DEFAULT_POLICY_TEMPLATE.

    Args:
        provider_id: Internal provider ID.
        plan_id: Internal plan ID.

    Returns:
        Comprehensive policy text for the LLM prompt.
    """
    policy = resolve_policy(provider_id, plan_id)
    if not policy:
        return (
            "POLICY CONTEXT UNAVAILABLE\n"
            "The user's insurance provider or plan could not be identified. "
            "Coverage assessment cannot be performed without policy details. "
            "Recommend the claimant contact their insurer directly."
        )

    coverage = policy.get("coverage", {})
    lines = []

    # Header
    lines.append(f"INSURANCE POLICY: {policy.get('plan_name', 'Unknown')}")
    lines.append(f"Provider: {policy.get('provider_name', 'Unknown')}")
    lines.append(f"Claims Phone: {policy.get('claims_phone', 'N/A')}")
    lines.append(f"Claims URL: {policy.get('claims_url', 'N/A')}")
    lines.append("=" * 60)

    # Coverage details
    lines.append("\nCOVERAGE DETAILS:")
    lines.append("-" * 40)

    col = coverage.get("collision", {})
    lines.append(f"\n* Collision Coverage: {'INCLUDED' if col.get('included') else 'NOT INCLUDED'}")
    if col.get("included"):
        lines.append(f"  Description: {col.get('description', 'N/A')}")
        lines.append(f"  Coverage Limit: {col.get('limit', 'N/A')}")
        lines.append(f"  Deductible: ${col.get('deductible', 'N/A')} per incident")

    comp = coverage.get("comprehensive", {})
    lines.append(f"\n* Comprehensive Coverage: {'INCLUDED' if comp.get('included') else 'NOT INCLUDED'}")
    if comp.get("included"):
        lines.append(f"  Description: {comp.get('description', 'N/A')}")
        lines.append(f"  Coverage Limit: {comp.get('limit', 'N/A')}")
        lines.append(f"  Deductible: ${comp.get('deductible', 'N/A')} per incident")

    liab = coverage.get("liability", {})
    if liab:
        lines.append(f"\n* Liability Coverage:")
        lines.append(f"  Bodily Injury: ${liab.get('bodily_injury_per_person', 0):,} per person / ${liab.get('bodily_injury_per_accident', 0):,} per accident")
        lines.append(f"  Property Damage: ${liab.get('property_damage', 0):,} per accident")

    rental = coverage.get("rental_reimbursement", {})
    lines.append(f"\n* Rental Reimbursement: {'INCLUDED' if rental.get('included') else 'NOT INCLUDED'}")
    if rental.get("included"):
        lines.append(f"  Daily Limit: ${rental.get('daily_limit', 0)}/day")
        lines.append(f"  Maximum Duration: {rental.get('max_days', 0)} days")
        if rental.get("description"):
            lines.append(f"  Details: {rental['description']}")

    roadside = coverage.get("roadside_assistance", {})
    lines.append(f"\n* Roadside Assistance: {'INCLUDED' if roadside.get('included') else 'NOT INCLUDED'}")
    if roadside.get("included"):
        lines.append(f"  Towing Limit: {roadside.get('towing_limit_miles', 'N/A')} miles")
        lines.append(f"  Services: {', '.join(roadside.get('services', []))}")

    glass = coverage.get("glass_coverage", {})
    lines.append(f"\n* Glass Coverage: {'INCLUDED' if glass.get('included') else 'NOT INCLUDED'}")
    if glass.get("included"):
        lines.append(f"  Deductible: ${glass.get('deductible', 'N/A')}")
    if glass.get("description"):
        lines.append(f"  Details: {glass['description']}")

    um = coverage.get("uninsured_motorist", {})
    if um.get("included"):
        lines.append(f"\n* Uninsured/Underinsured Motorist: INCLUDED")
        lines.append(f"  Bodily Injury: ${um.get('bodily_injury_per_person', 0):,} per person / ${um.get('bodily_injury_per_accident', 0):,} per accident")

    # Exclusions
    exclusions = policy.get("exclusions", [])
    lines.append("\n" + "=" * 60)
    lines.append("POLICY EXCLUSIONS:")
    lines.append("-" * 40)
    for i, exc in enumerate(exclusions, 1):
        lines.append(f"  {i}. {exc}")

    # Claim conditions
    conditions = policy.get("claim_conditions", [])
    lines.append("\n" + "=" * 60)
    lines.append("CLAIM FILING CONDITIONS:")
    lines.append("-" * 40)
    for i, cond in enumerate(conditions, 1):
        lines.append(f"  {i}. {cond}")

    # Payment and depreciation
    lines.append("\n" + "=" * 60)
    lines.append("VALUATION AND PAYMENT:")
    lines.append("-" * 40)
    lines.append(f"  Depreciation Method: {policy.get('depreciation_method', 'N/A')}")
    lines.append(f"  Payment Terms: {policy.get('payment_terms', 'N/A')}")

    return "\n".join(lines)