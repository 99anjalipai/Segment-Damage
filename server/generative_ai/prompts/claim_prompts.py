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
Your job is to write a complete claim report that the claimant can submit directly to their insurance company.

IMPORTANT RULES:
- The claim letter MUST be written FROM the claimant TO the insurance company. Use first person ("I am writing...", "My vehicle...").
- Use ONLY the information provided below. Do NOT make up any facts.
- If any information is missing, write [PLACEHOLDER] and note it in the Missing Information section.
- Keep the language professional but clear.

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
OUTPUT INSTRUCTIONS
===============================================================================
Write the following four sections. Do not skip any section.

### 1. Insurance Claim Letter

Write a formal letter FROM the claimant TO the insurance company. This letter is written in FIRST PERSON by the vehicle owner. It should be ready to send without any edits.

Structure:
- Header: Claimant's name, address, phone, email, today's date, insurance company name, policy number, and a subject line (e.g., "Subject: Insurance Claim for Vehicle Damage - Policy #12345")
- Opening: "Dear Claims Department," followed by one sentence stating the purpose (e.g., "I am writing to file a claim for damage to my vehicle...")
- Incident description: What happened, when, and where. Keep it factual and concise. Use the claimant's own words from the event description.
- Damage summary: Reference the damage found by the AI image analysis. Do not add damage that was not detected.
- Request: Ask the insurance company to open a claim and process the request. Mention that photos and documentation are attached.
- Closing: "Sincerely," followed by the claimant's name and contact info.

### 2. Damage Analysis

For each image analyzed, provide:
- The affected area using standard automotive terms (e.g., "front bumper", "left rear quarter panel", "windshield")
- Damage type: dent, scratch, crack, deformation, paint transfer, shatter, or other
- Severity: Minor, Moderate, or Severe (with a brief reason)
- Why this damage is relevant to the claim

Only describe damage that appears in the AI analysis above. Do not invent additional damage.

### 3. Coverage Assessment

Using ONLY the policy context provided above, assess:
- Which damages are likely covered and why
- Any exclusions or limitations that may apply
- Deductible amount and out-of-pocket costs (if stated in the policy; otherwise write [PLACEHOLDER])
- Overall confidence: High, Medium, or Low

If no policy context was provided, state that a coverage assessment cannot be made without policy details.

### 4. Your Action Plan

Write a short personalized checklist addressed directly to the claimant using "you" language. These are the steps they should take in the next 24-72 hours based on this specific incident. For example:
- "File a police report at [location] if you haven't already"
- "Call [insurance company] at their claims hotline to submit this letter"
- "Get a repair estimate from an approved body shop"
- "Arrange a rental car if your vehicle is not safe to drive"
- "Save all receipts related to the accident (towing, rental, medical)"

Keep it to 4-6 items. Be specific to the incident, not generic.
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