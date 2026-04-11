import streamlit as st
import numpy as np
import time
import os
from PIL import Image
from datetime import datetime

# Import services
from services.segmentation import segment_damage, overlay_mask
from services.damage_analyzer import generate_detected_damage, analyze_masks
from services.policy_engine import (
    get_provider_list, get_plan_list, resolve_policy,
    generate_policy_context, format_coverage_summary, get_provider_contact,
)
from services.repair_estimator import get_estimate_summary_for_llm
from services.pdf_generator import generate_claim_pdf, REPORTLAB_AVAILABLE
from generative_ai.core.claim_drafter import ClaimDraftCore

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoClaim AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0F172A 0%, #1E40AF 40%, #3B82F6 70%, #60A5FA 100%);
    z-index: 9999;
}

section[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F8FAFC !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 0.6rem 1.8rem;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(30,64,175,0.25);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(30,64,175,0.35);
}

.report-section {
    background: #FAFBFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #1E40AF;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.report-section-green {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.report-section-amber {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.report-section-purple {
    background: #FAF5FF;
    border: 1px solid #E9D5FF;
    border-left: 4px solid #7C3AED;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

.metric-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: white;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60A5FA;
    line-height: 1.2;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background: #DCFCE7; color: #166534; }
.badge-info { background: #DBEAFE; color: #1E40AF; }

.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 0.8rem;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    font-weight: 500;
}
.step-indicator .step-num {
    background: #1E40AF;
    color: white;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    flex-shrink: 0;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #065F46 0%, #10B981 100%) !important;
    box-shadow: 0 2px 8px rgba(6,95,70,0.25);
}
.stDownloadButton > button:hover {
    box-shadow: 0 4px 16px rgba(6,95,70,0.35);
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────

def get_segmentation_models():
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    models = []
    if os.path.exists(outputs_dir):
        for folder in sorted(os.listdir(outputs_dir)):
            fp = os.path.join(outputs_dir, folder)
            if os.path.isdir(fp) and os.path.exists(os.path.join(fp, "best.pt")):
                models.append(folder)
    return models

def render_metric(value, label):
    st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

def render_step(num, text):
    st.markdown(f'<div class="step-indicator"><div class="step-num">{num}</div>{text}</div>', unsafe_allow_html=True)


# ─── Main ────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚡ AutoClaim AI")
        st.caption("Intelligent damage assessment & claims automation")
        st.divider()

        st.markdown("#### AI model")
        models = get_segmentation_models()
        seg_model = st.selectbox("Segmentation model", models, index=0 if models else None, help="Trained CV model for damage detection")

        st.divider()
        st.markdown("#### Pipeline")
        render_step(1, "Upload damage photos")
        render_step(2, "AI segments damage")
        render_step(3, "Analyze & estimate cost")
        render_step(4, "Generate claim report")
        render_step(5, "Download PDF")
        st.divider()
        st.caption("UNet · YOLOv8 · Qwen2-VL · LangChain")

    # Header
    st.markdown("# File your claim in minutes, not days.")
    st.markdown("Upload photos of your damaged vehicle. Our AI analyzes damage, cross-references your policy, estimates repairs, and drafts a submission-ready claim.")
    st.markdown("---")

    # ── Form: Row 1 ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Your information")
        user_name = st.text_input("Full name", placeholder="Jane Doe")
        c1, c2 = st.columns(2)
        with c1:
            user_phone = st.text_input("Phone", placeholder="(555) 123-4567")
        with c2:
            user_email = st.text_input("Email", placeholder="jane@email.com")
        user_address = st.text_input("Address", placeholder="123 Main St, Boston, MA 02101")

    with col_r:
        st.markdown("#### Insurance details")
        providers = get_provider_list()
        provider_names = [p["display_name"] for p in providers]
        provider_ids = [p["id"] for p in providers]
        sel_prov_idx = st.selectbox("Insurance provider", range(len(providers)), format_func=lambda i: provider_names[i])
        sel_prov_id = provider_ids[sel_prov_idx]
        insurance_company = provider_names[sel_prov_idx]

        plans = get_plan_list(sel_prov_id)
        plan_names = [p["plan_name"] for p in plans]
        plan_ids = [p["id"] for p in plans]
        sel_plan_idx = st.selectbox("Plan tier", range(len(plans)), format_func=lambda i: plan_names[i]) if plans else 0
        sel_plan_id = plan_ids[sel_plan_idx] if plans else "default"

        policy_number = st.text_input("Policy number", placeholder="POL-2026-XXXXXX")

        resolved = resolve_policy(sel_prov_id, sel_plan_id)
        deductible = resolved.get("coverage", {}).get("collision", {}).get("deductible", 500) if resolved else 500
        if resolved:
            with st.expander("View your coverage summary"):
                st.code(format_coverage_summary(resolved), language=None)

    # ── Form: Row 2 ──────────────────────────────────────────────
    st.markdown("---")
    col_v, col_i = st.columns(2)

    with col_v:
        st.markdown("#### Vehicle")
        v1, v2, v3 = st.columns(3)
        with v1:
            vehicle_year = st.text_input("Year", placeholder="2024")
        with v2:
            vehicle_make = st.text_input("Make", placeholder="Toyota")
        with v3:
            vehicle_model = st.text_input("Model", placeholder="Camry")
        vi1, vi2 = st.columns(2)
        with vi1:
            vehicle_vin = st.text_input("VIN", placeholder="1HGBH41JXMN109186")
        with vi2:
            license_plate = st.text_input("License plate", placeholder="ABC-1234")

    with col_i:
        st.markdown("#### Incident")
        d1, d2 = st.columns(2)
        with d1:
            incident_date = st.date_input("Date")
        with d2:
            incident_time = st.text_input("Time", placeholder="3:15 PM")
        incident_location = st.text_input("Location", placeholder="Main St & Elm Ave, Roxbury, MA")
        event_description = st.text_area("What happened?", placeholder="Describe the accident in detail...", height=120)

    # ── Photos ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Damage photos")
    st.caption("Upload clear photos of all damaged areas from multiple angles.")
    uploaded_files = st.file_uploader("Drop images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")

    # ── Validation ───────────────────────────────────────────────
    required = [user_name, user_phone, user_email, user_address, insurance_company, policy_number, vehicle_year, vehicle_make, vehicle_model, vehicle_vin, license_plate, incident_time, incident_location, event_description]
    all_filled = all(str(x).strip() for x in required)
    has_imgs = uploaded_files is not None and len(uploaded_files) > 0

    if has_imgs and not all_filled:
        st.warning("Please complete all fields before generating your claim.")

    if has_imgs and all_filled:
        images = [Image.open(f) for f in uploaded_files]
        img_cols = st.columns(min(len(images), 4))
        for idx, img in enumerate(images):
            with img_cols[idx % len(img_cols)]:
                st.image(img, use_container_width=True, caption=f"Photo {idx+1}")

        st.markdown("")
        if st.button("⚡  Analyze damage & generate claim", use_container_width=True):

            # Stage 1: Segmentation
            from services import segmentation as seg
            masks, overlayed_images = [], []
            with st.spinner("Running AI damage detection..."):
                prog = st.progress(0)
                for idx, img in enumerate(images):
                    mask = seg.segment_damage(np.array(img), model_name=seg_model)
                    ov = seg.overlay_mask(img, mask, color=(255, 50, 50), alpha=0.6)
                    masks.append(mask)
                    overlayed_images.append(ov)
                    prog.progress((idx + 1) / len(images))
                prog.empty()

            st.markdown("##### Detection results")
            det_cols = st.columns(min(len(images), 4))
            for idx in range(len(images)):
                with det_cols[idx % len(det_cols)]:
                    st.image(overlayed_images[idx], use_container_width=True, caption=f"Detection {idx+1}")

            # Stage 2: Analysis
            damage_text = generate_detected_damage(masks)
            damage_analyses = analyze_masks(masks)
            policy_context = generate_policy_context(sel_prov_id, sel_plan_id)
            policy_summary = format_coverage_summary(resolved) if resolved else ""
            provider_contact = get_provider_contact(sel_prov_id)
            repair_text, repair_result = get_estimate_summary_for_llm(damage_analyses, deductible=deductible)

            # Metrics
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1: render_metric(str(len(images)), "Images analyzed")
            with m2: render_metric(str(repair_result["num_regions"]), "Damage regions")
            with m3: render_metric(f"${repair_result['total_midpoint']:,}", "Est. repair cost")
            with m4: render_metric(f"${min(deductible, repair_result['total_midpoint']):,}", f"Deductible (${deductible})")

            # Stage 3: LLM Generation
            st.markdown("---")
            st.markdown("##### Generating claim report...")

            if "drafter" not in st.session_state:
                st.session_state.drafter = ClaimDraftCore(provider="qwen-vl")
            llm = st.session_state.drafter.llm
            overlayed_pil = [Image.fromarray(ov) for ov in overlayed_images]

            claimant_info = {"user_name": user_name, "user_address": user_address, "user_phone": user_phone, "user_email": user_email}
            vehicle_info_d = {"vehicle_year": vehicle_year, "vehicle_make": vehicle_make, "vehicle_model": vehicle_model, "vehicle_vin": vehicle_vin, "license_plate": license_plate}
            incident_info_d = {"incident_date": str(incident_date), "incident_time": incident_time, "incident_location": incident_location, "event_description": event_description}
            insurance_info_d = {"insurance_company": insurance_company, "policy_number": policy_number}

            # Prompt 1
            with st.status("Writing claim letter...", expanded=False) as s:
                p1 = f"""Write a formal insurance claim letter written by {user_name} to {insurance_company}.

RULES: First person as {user_name}. No technical data (bounding boxes, percentages). No invented facts. You are the claimant, not the insurer.

CLAIMANT: {user_name}, {user_address}, Phone: {user_phone}, Email: {user_email}
INSURANCE: {insurance_company}, Policy: {policy_number}
VEHICLE: {vehicle_year} {vehicle_make} {vehicle_model}, VIN: {vehicle_vin}, Plate: {license_plate}
INCIDENT: {incident_date} at {incident_time}, {incident_location}
WHAT HAPPENED: {event_description}
DAMAGE (AI): {damage_text}

FORMAT:
{user_name}
{user_address}
Phone: {user_phone} | Email: {user_email}
Date: {incident_date}

{insurance_company} - Claims Department
Policy: {policy_number}
Subject: Insurance Claim for Vehicle Damage

Dear Claims Department,
[File a claim. Describe incident. Summarize damage in plain English. Mention attached photos. Request claim be opened.]

Sincerely,
{user_name}
{user_phone} | {user_email}"""
                section1 = llm.generate_raw(p1, images=overlayed_pil, max_tokens=1000)
                s.update(label="Claim letter ready", state="complete")

            # Prompt 2
            with st.status("Analyzing damage...", expanded=False) as s:
                p2 = f"""Automotive damage assessor. Attached images show damage in red.

AI DATA: {damage_text}
REPAIR ESTIMATE: {repair_text}

Write a report for ALL {repair_result['num_regions']} damage regions. For each:
1. Affected area (automotive terms)
2. Damage type: dent/scratch/crack/deformation/shatter/structural
3. Severity with explanation
4. Estimated repair cost range
5. Claim relevance

Cover ALL regions. Do NOT stop after one."""
                section2 = llm.generate_raw(p2, images=overlayed_pil, max_tokens=1000)
                s.update(label="Damage analysis ready", state="complete")

            # Prompt 3
            with st.status("Assessing coverage...", expanded=False) as s:
                p3 = f"""Insurance policy analyst. Read the policy CAREFULLY.

POLICY: {policy_context}
DAMAGES: {damage_text}
REPAIR ESTIMATE: ${repair_result['total_low']:,} - ${repair_result['total_high']:,}

Answer using ONLY the policy:
1. COVERED ITEMS: Which damages are covered? Reference policy language.
2. EXCLUSIONS: Only those explicitly stated. If none apply, say so.
3. DEDUCTIBLE: Quote exact amount. Calculate insurer pays = estimate - deductible.
4. RENTAL COVERAGE: Included? Daily/duration limits?
5. CLAIM FILING: Deadlines and requirements?
6. CONFIDENCE: High/Medium/Low with justification.

Do NOT invent policy terms."""
                section3 = llm.generate_raw(p3, max_tokens=700)
                s.update(label="Coverage assessment ready", state="complete")

            # Prompt 4
            with st.status("Creating action plan...", expanded=False) as s:
                claims_phone = provider_contact.get("claims_phone", "")
                p4 = f"""5 action items for {user_name} after their accident.

Context: Accident {incident_date} at {incident_location}. Insurer: {insurance_company} (#{policy_number}). Claims phone: {claims_phone}. Vehicle: {vehicle_year} {vehicle_make} {vehicle_model}. Repair estimate: ${repair_result['total_midpoint']:,}. Deductible: ${deductible}.

Numbered list only. No greeting/sign-off. "You" language. Mention {insurance_company} by name."""
                section4 = llm.generate_raw(p4, max_tokens=400)
                s.update(label="Action plan ready", state="complete")

            # ── Report Display ───────────────────────────────────
            st.markdown("---")
            st.markdown("## Your claim report")
            plan_label = plan_names[sel_plan_idx] if plans else "Custom"
            st.markdown(
                f'<span class="status-badge badge-success">AI generated</span> '
                f'<span class="status-badge badge-info">{insurance_company} · {plan_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            st.markdown("#### 📄 Claim letter")
            st.markdown(section1)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="report-section-amber">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Damage analysis")
            st.markdown(section2)
            with st.expander("Detailed cost breakdown"):
                st.code(repair_text, language=None)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="report-section-purple">', unsafe_allow_html=True)
            st.markdown("#### 📋 Coverage assessment")
            st.markdown(section3)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="report-section-green">', unsafe_allow_html=True)
            st.markdown("#### ✅ Your action plan")
            st.markdown(section4)
            if claims_phone:
                st.info(f"**{insurance_company} claims hotline:** {claims_phone}")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── PDF / Downloads ──────────────────────────────────
            st.markdown("---")
            if REPORTLAB_AVAILABLE:
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_path = generate_claim_pdf(
                            sections={"claim_letter": section1, "damage_analysis": section2, "coverage_assessment": section3, "action_plan": section4},
                            claimant_info=claimant_info,
                            vehicle_info=vehicle_info_d,
                            incident_info=incident_info_d,
                            insurance_info=insurance_info_d,
                            original_images=images,
                            overlayed_images=overlayed_images,
                            repair_estimate_text=repair_text,
                            policy_summary=policy_summary,
                        )
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        d1, d2, _ = st.columns([1, 1, 2])
                        with d1:
                            st.download_button("⬇  Download full report (PDF)", pdf_bytes, f"autoclaim_{user_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf", use_container_width=True)
                        with d2:
                            st.download_button("📄  Download letter only", section1, f"claim_letter_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain", use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
            else:
                st.info("Install `reportlab` for PDF export: `pip install reportlab`")

            st.caption("**Disclaimer:** AI-generated draft. Review before submission. Estimates are approximate. Consult a licensed adjuster for final evaluation.")


if __name__ == "__main__":
    main()