from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE

class BaseLLMClient:
    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
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

    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
        # If images/masks are provided and the model supports vision, pass them (future extension)
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

    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
        response = self.chain.invoke(prompt_vars)
        return response.content

# Qwen-VL (Vision-Language) integration (using Transformers pipeline)
class QwenVLLLMClient(BaseLLMClient):
    def __init__(self, model_name):
        # We handle text generation for Qwen-VL using Huggingface
        # If Qwen-VL-Chat model is available on machine or huggingface:
        from transformers import pipeline
        import torch
        try:
            # Force CPU for the language model to prevent MPS memory allocation limits/crashes
            # MPS on some Macs crashes when allocating >4GB continuously for LLM temp tensors
            self.pipe = pipeline("text-generation", model=model_name, max_new_tokens=1000, device=torch.device("cpu"))
        except Exception:
            # Fallback to tiny models if memory is tight for demonstration
            self.pipe = pipeline("text-generation", model="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=500, device=torch.device("cpu"))
    
    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
        # If images/masks are provided and the pipeline supports them, pass them (future extension)
        from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE
        try:
            prompt = CLAIM_DRAFT_PROMPT_TEMPLATE.format(**prompt_vars)
        except KeyError:
            # Safely build prompt if strict template fails due to missing keys in Qwen mode
            prompt = f"Write an insurance claim for this incident: {prompt_vars.get('event_description')} with damage: {prompt_vars.get('detected_damage')}"
        
        result = self.pipe(prompt, max_new_tokens=512)
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
