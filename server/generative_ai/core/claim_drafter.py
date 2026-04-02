import os
from langchain_google_genai import ChatGoogleGenerativeAI
from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE

DEFAULT_POLICY_TEMPLATE = """
STANDARD AUTO INSURANCE POLICY - COVERAGE SUMMARY
* Comprehensive & Collision Coverage: Included.
* Deductible: $500 per incident.
* Rental Reimbursement: Up to $30/day for 30 days.
* Conditions: The policy covers accidental damage, including collisions with other vehicles, stationary objects, and weather-related damage. The insured must report the incident to the claims department and provide photographic evidence of the damage.
"""


from generative_ai.core.llm_clients import get_llm_client

class ClaimDraftCore:
    def __init__(self, provider="gemini", api_key=None, model_name=None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name or self.default_model(provider)
        self.llm = get_llm_client(self.provider, self.api_key, self.model_name)

    @staticmethod
    def default_model(provider):
        if provider == "gemini":
            return "models/gemini-2.5-flash"
        elif provider == "openai":
            return "gpt-3.5-turbo"
        # Add more defaults as needed
        return None

    def generate_draft(self, detected_damage: str, event_description: str, insurance_context: str = None) -> str:
        if not insurance_context:
            insurance_context = DEFAULT_POLICY_TEMPLATE
        prompt_vars = {
            "detected_damage": detected_damage,
            "event_description": event_description,
            "insurance_context": insurance_context
        }
        return self.llm.generate(prompt_vars)
