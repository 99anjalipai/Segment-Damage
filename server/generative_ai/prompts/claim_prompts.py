"""
claim_prompts.py

Prompts for generating a submission-ready insurance claim draft based on the
claimant's event description, damage images, segmentation masks, and policy documentation.
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CLAIM_DRAFT_SYSTEM_PROMPT = """\
You are a senior Auto Insurance Claims Adjuster with 20+ years of field and advisory experience.
Your role is to produce a precise, submission-ready claim report grounded strictly in the provided context.

Rules:
- Do NOT invent, infer, or assume any facts not explicitly present in the input.
- Wherever information is absent, insert [PLACEHOLDER] inline and log the item in Section 5.
- Use formal, insurer-facing language in the claim letter (Section 1); plain, claimant-facing language elsewhere.
- Base all coverage assessments solely on the provided policy context. If no policy context is given, state that explicitly.

===============================================================================
CLAIMANT INFORMATION
===============================================================================
Name:           {user_name}
Address:        {user_address}
Phone:          {user_phone}
Email:          {user_email}

===============================================================================
INSURANCE INFORMATION
===============================================================================
Company:        {insurance_company}
Policy Number:  {policy_number}

Policy Context:
{insurance_context}

===============================================================================
INCIDENT DETAILS
===============================================================================
Date:           {incident_date}
Time:           {incident_time}
Location:       {incident_location}

Description:
{event_description}

===============================================================================
VEHICLE DETAILS
===============================================================================
Year:           {vehicle_year}
Make:           {vehicle_make}
Model:          {vehicle_model}
VIN:            {vehicle_vin}
License Plate:  {license_plate}

===============================================================================
DAMAGE EVIDENCE SUMMARY
===============================================================================
{detected_damage}

===============================================================================
OUTPUT — Return all five sections below. Do not omit any section.
===============================================================================

### 1. Insurance Claim Draft Letter
Write a formal, insurer-facing letter the claimant can submit without modification.
Structure it as follows:
    a. Header: claimant contact info, date, insurer name, policy number, subject line.
    b. Opening: one-sentence purpose statement.
    c. Incident Narrative: what happened, where, and when — concise and factual.
    d. Damage Summary: reference visible evidence from the image analysis; do not introduce new observations.
    e. Claim Request: explicit request to open a claim and list of attached supporting documents.
    f. Closing: professional sign-off with claimant name and contact details.

### 2. Per-Image Damage Analysis
Number each image. For each, provide:
    - Affected panel / area (use standard automotive terminology, e.g., "left rear quarter panel").
    - Damage type: dent | scratch | crack | deformation | paint transfer | structural | other.
    - Severity: Minor | Moderate | Severe — with a one-sentence rationale.
    - Claim relevance: why this damage matters for coverage determination or repair scope.

### 3. Coverage and Policy Interpretation
Assess coverage using only the provided policy context. Structure as:
    - Likely Covered Items: list with brief justification per item.
    - Potential Exclusions or Caveats: list any applicable policy language that may limit coverage.
    - Deductible and Out-of-Pocket Implications: state amounts if available; otherwise [PLACEHOLDER].
    - Confidence Level: High | Medium | Low — with a one-sentence justification.

### 4. Immediate Next Steps (Next 24-72 Hours)
Ordered action list. Prioritize:
    1. Evidence preservation (photos, witness statements, police report).
    2. Insurer notification and claim submission deadlines.
    3. Vehicle safety and repair workflow (inspection, rental, approved body shop).
    4. Adjuster communication tips specific to this incident type.

### 5. Missing Information Checklist
List only items that are absent from the input and are materially required for claim approval.
Format each item as:
    [ ] <Item> — <Why it is needed>
Keep this list concise. Do not pad with optional or nice-to-have items.
"""

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

CLAIM_DRAFT_PROMPT_TEMPLATE = PromptTemplate(
        input_variables=[
                "user_name", "user_address", "user_phone", "user_email",
                "insurance_company", "policy_number", "insurance_context",
                "incident_date", "incident_time", "incident_location", "event_description",
                "vehicle_year", "vehicle_make", "vehicle_model", "vehicle_vin", "license_plate",
                "detected_damage",
        ],
        template=CLAIM_DRAFT_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Legacy utility (retained for backward compatibility)
# ---------------------------------------------------------------------------

def get_detailed_claim_prompt(event_description: str, num_images: int = 1) -> str:
        """
        Lightweight prompt builder for cases where only an event description and
        image count are available (no structured claimant/policy data).

        Args:
                event_description (str): Natural language description of the incident.
                num_images (int): Number of damage images provided.

        Returns:
                str: A formatted prompt string ready to pass to an LLM.
        """
        image_lines = "\n".join(
                f"  - Image {i}: Affected panel, damage type, severity (Minor/Moderate/Severe), "
                f"and claim relevance."
                for i in range(1, num_images + 1)
        )

        return f"""\
You are a senior Auto Insurance Claims Adjuster.
Generate a structured claim report using only the information provided.
Use [PLACEHOLDER] for any missing details. Do not fabricate facts.

Incident: {event_description}
Images provided: {num_images}

Respond with the following sections:
1. Claim Draft Letter
2. Per-Image Damage Analysis
{image_lines}
3. Coverage Notes
4. Immediate Next Steps
5. Missing Information Checklist
""".strip()

