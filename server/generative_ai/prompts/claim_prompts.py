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
You are an AI assistant helping a vehicle owner draft an insurance claim after an accident.

CRITICAL RULES:
1. The claim letter is written FROM the claimant TO the insurance company. Always use first person ("I", "my vehicle").
2. Use ONLY the facts provided below. If a detail is not in the input, do NOT invent it. Write [PLACEHOLDER] instead.
3. For coverage assessment, refer ONLY to the policy text provided. Do NOT make up policy terms, exclusions, or conditions.
4. You MUST output exactly four clearly separated sections using the exact headers shown below.
5. In the closing of the claim letter, use the actual claimant name, address, phone, and email provided below -- not generic placeholders.

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
DAMAGE DETECTED BY AI IMAGE ANALYSIS
===============================================================================
{detected_damage}

===============================================================================
OUTPUT FORMAT -- You MUST use these exact four section headers, each on its own line.
===============================================================================

---SECTION 1: CLAIM LETTER---

Write a formal letter FROM {user_name} TO {insurance_company}. Written in first person.

Format:
{user_name}
{user_address}
Phone: {user_phone}
Email: {user_email}
Date: {incident_date}

To: {insurance_company}
Policy Number: {policy_number}
Subject: Insurance Claim for Vehicle Damage - Policy #{policy_number}

Dear Claims Department,

[Write the body of the letter here. Include:
- One sentence stating purpose: "I am writing to file a claim for damage to my {vehicle_year} {vehicle_make} {vehicle_model}..."
- Incident narrative: restate ONLY what is in the event description above. Do not add details.
- Damage found: reference ONLY the damage from the AI analysis section. Do not add extra damage.
- Request to open a claim and note that photos are attached.]

Sincerely,
{user_name}
{user_address}
Phone: {user_phone}
Email: {user_email}

---SECTION 2: DAMAGE ANALYSIS---

For each damaged area found in the AI image analysis above, write:
- Affected area (use automotive terms like "front bumper", "left rear quarter panel")
- Damage type: dent | scratch | crack | deformation | paint transfer | shatter | other
- Severity: Minor | Moderate | Severe (one sentence reason)
- Claim relevance: why this matters for the claim

ONLY describe damage that is in the AI analysis. Do NOT add any damage not listed above.

---SECTION 3: COVERAGE ASSESSMENT---

Using ONLY the policy context above, state:
- Which damages are likely covered (quote the relevant policy language)
- Any exclusions that apply (ONLY if explicitly stated in the policy text)
- Deductible: state the amount from the policy, or [PLACEHOLDER] if not provided
- Confidence: High | Medium | Low

WARNING: Do NOT invent policy terms. If the policy text does not mention an exclusion, do not make one up. Only reference what is explicitly written in the policy context.

---SECTION 4: YOUR ACTION PLAN---

Write 4-6 action items addressed to the claimant using "you" language. Make them specific to this incident:
- Reference the actual insurance company name, incident location, etc.
- Include concrete steps like filing a police report, calling the insurer, getting a repair estimate
- Do not include generic filler items
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
You are an AI assistant helping a vehicle owner draft an insurance claim.
Write a claim report using only the information provided.
The claim letter must be written FROM the claimant TO the insurance company in first person.
Use [PLACEHOLDER] for any missing details. Do not make up facts.

Incident: {event_description}
Images provided: {num_images}

Respond with the following sections:
1. Claim Letter (written by the claimant in first person)
2. Damage Analysis
{image_lines}
3. Coverage Assessment
4. Your Action Plan (addressed to the claimant using "you" language)
""".strip()