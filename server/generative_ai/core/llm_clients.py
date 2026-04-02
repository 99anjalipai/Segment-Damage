from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE

class BaseLLMClient:
    def generate(self, prompt_vars: dict) -> str:
        raise NotImplementedError("Subclasses must implement generate()")

class GeminiLLMClient(BaseLLMClient):
    def __init__(self, api_key, model_name):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=api_key
        )
        self.chain = CLAIM_DRAFT_PROMPT_TEMPLATE | self.llm

    def generate(self, prompt_vars: dict) -> str:
        response = self.chain.invoke(prompt_vars)
        return response.content

# Example stub for OpenAI (GPT-3.5/4)
class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key, model_name):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=api_key
        )
        self.chain = CLAIM_DRAFT_PROMPT_TEMPLATE | self.llm

    def generate(self, prompt_vars: dict) -> str:
        response = self.chain.invoke(prompt_vars)
        return response.content

# Qwen-VL (Vision-Language) integration (using Transformers pipeline)
class QwenVLLLMClient(BaseLLMClient):
    def __init__(self, model_name):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model_name)
        # For real VLM, use pipeline("image-to-text", ...) or similar if available

    def generate(self, prompt_vars: dict) -> str:
        # Compose prompt from template
        prompt = CLAIM_DRAFT_PROMPT_TEMPLATE.format(**prompt_vars)
        result = self.pipe(prompt, max_new_tokens=512)
        # Huggingface pipeline returns a list of dicts
        return result[0]["generated_text"] if isinstance(result, list) else str(result)

def get_llm_client(provider: str, api_key: str, model_name: str):
    if provider == "qwen-vl":
        return QwenVLLLMClient(model_name)
    if provider == "gemini":
        return GeminiLLMClient(api_key, model_name)
    elif provider == "openai":
        return OpenAILLMClient(api_key, model_name)
    # Add more providers here
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
