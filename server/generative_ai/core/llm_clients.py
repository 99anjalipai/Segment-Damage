# from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE

# class BaseLLMClient:
#     def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
#         raise NotImplementedError("Subclasses must implement generate()")

# class GeminiLLMClient(BaseLLMClient):
#     def __init__(self, api_key, model_name):
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         self.llm = ChatGoogleGenerativeAI(
#             model=model_name,
#             temperature=0.2,
#             google_api_key=api_key
#         )
#         self.chain = CLAIM_DRAFT_PROMPT_TEMPLATE | self.llm

#     def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
#         # If images/masks are provided and the model supports vision, pass them (future extension)
#         response = self.chain.invoke(prompt_vars)
#         return response.content

# # Example stub for OpenAI (GPT-3.5/4)
# class OpenAILLMClient(BaseLLMClient):
#     def __init__(self, api_key, model_name):
#         from langchain_openai import ChatOpenAI
#         self.llm = ChatOpenAI(
#             model=model_name,
#             temperature=0.2,
#             openai_api_key=api_key
#         )
#         self.chain = CLAIM_DRAFT_PROMPT_TEMPLATE | self.llm

#     def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
#         response = self.chain.invoke(prompt_vars)
#         return response.content

# # Qwen-VL (Vision-Language) integration (using Transformers pipeline)
# class QwenVLLLMClient(BaseLLMClient):
#     def __init__(self, model_name):
#         # We handle text generation for Qwen-VL using Huggingface
#         # If Qwen-VL-Chat model is available on machine or huggingface:
#         from transformers import pipeline
#         import torch
#         try:
#             # Force CPU for the language model to prevent MPS memory allocation limits/crashes
#             # MPS on some Macs crashes when allocating >4GB continuously for LLM temp tensors
#             self.pipe = pipeline("text-generation", model=model_name, max_new_tokens=1000, device=torch.device("cpu"))
#         except Exception:
#             # Fallback to tiny models if memory is tight for demonstration
#             self.pipe = pipeline("text-generation", model="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=500, device=torch.device("cpu"))
    
#     def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
#         # If images/masks are provided and the pipeline supports them, pass them (future extension)
#         from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE
#         try:
#             prompt = CLAIM_DRAFT_PROMPT_TEMPLATE.format(**prompt_vars)
#         except KeyError:
#             # Safely build prompt if strict template fails due to missing keys in Qwen mode
#             prompt = f"Write an insurance claim for this incident: {prompt_vars.get('event_description')} with damage: {prompt_vars.get('detected_damage')}"

#         # Add an explicit anchor so completion-style models begin at Section 1.
#         anchored_prompt = (
#             f"{prompt}\n\n"
#             "Start your response now. Begin exactly with:\n"
#             "### 1. Insurance Claim Draft Letter\n"
#         )

#         def _run_generation(input_prompt: str, token_budget: int):
#             try:
#                 return self.pipe(
#                     input_prompt,
#                     max_new_tokens=token_budget,
#                     return_full_text=False,
#                     do_sample=False,
#                 )
#             except TypeError:
#                 # Backward-compatible fallback for older transformers pipeline signatures.
#                 return self.pipe(input_prompt, max_new_tokens=token_budget)

#         result = _run_generation(anchored_prompt, 900)
#         generated_text = result[0]["generated_text"] if isinstance(result, list) else str(result)

#         # Defensive cleanup: some text-generation pipelines still echo input text.
#         if generated_text.startswith(anchored_prompt):
#             generated_text = generated_text[len(anchored_prompt):].lstrip()

#         # Retry once if output is malformed/truncated (e.g., only "6." or missing Section 1 heading).
#         if "### 1. Insurance Claim Draft Letter" not in generated_text or len(generated_text.strip()) < 120:
#             retry_prompt = (
#                 f"{anchored_prompt}\n"
#                 "Do not continue numbering from prior text. Start at Section 1 and include all five sections.\n"
#             )
#             retry_result = _run_generation(retry_prompt, 1100)
#             retry_text = retry_result[0]["generated_text"] if isinstance(retry_result, list) else str(retry_result)
#             if retry_text.startswith(retry_prompt):
#                 retry_text = retry_text[len(retry_prompt):].lstrip()
#             if len(retry_text.strip()) > len(generated_text.strip()):
#                 generated_text = retry_text

#         return generated_text

# def get_llm_client(provider: str, api_key: str, model_name: str):
#     if provider == "qwen-vl":
#         return QwenVLLLMClient(model_name)
#     if provider == "gemini":
#         return GeminiLLMClient(api_key, model_name)
#     elif provider == "openai":
#         return OpenAILLMClient(api_key, model_name)
#     # Add more providers here
#     else:
#         raise ValueError(f"Unknown LLM provider: {provider}")


from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE, CLAIM_DRAFT_SYSTEM_PROMPT

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
        response = self.chain.invoke(prompt_vars)
        return response.content


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


class QwenVLLLMClient(BaseLLMClient):
    """
    Qwen2-VL-2B-Instruct client for multimodal claim generation.
    Accepts segmented damage images alongside structured text input.
    Optimized for Apple Silicon (MPS) with CPU fallback.
    """

    def __init__(self, model_name=None):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_id = model_name or "Qwen/Qwen2-VL-2B-Instruct"

        # Force CPU -- MPS crashes on Qwen2-VL due to >4GB tensor allocation
        # in the vision encoder (MPSTemporaryNDArray > 2^32 bytes limit)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        print(f"[QwenVL] Loading {self.model_id} on {self.device} ({self.dtype})")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=None,
        ).to(self.device).eval()

        # Aggressive image token limits to keep memory and inference time
        # reasonable on a 16GB MacBook Air running on CPU
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=128 * 28 * 28,    # ~100K pixels min
            max_pixels=256 * 28 * 28,    # ~200K pixels max
        )

        print(f"[QwenVL] Model loaded successfully.")

    def _build_prompt_text(self, prompt_vars: dict) -> str:
        """
        Fill in the claim prompt template with available variables.
        Missing keys get replaced with [PLACEHOLDER].
        """
        expected_keys = [
            "user_name", "user_address", "user_phone", "user_email",
            "insurance_company", "policy_number", "insurance_context",
            "incident_date", "incident_time", "incident_location", "event_description",
            "vehicle_year", "vehicle_make", "vehicle_model", "vehicle_vin", "license_plate",
            "detected_damage",
        ]
        safe_vars = {k: prompt_vars.get(k, "[PLACEHOLDER]") for k in expected_keys}
        return CLAIM_DRAFT_SYSTEM_PROMPT.format(**safe_vars)

    def _prepare_image_content(self, images) -> list:
        """
        Convert image inputs (PIL Images or file paths) into Qwen2-VL
        message content blocks.
        """
        from PIL import Image
        import base64
        from io import BytesIO

        image_blocks = []
        if images is None:
            return image_blocks

        if not isinstance(images, list):
            images = [images]

        for img in images:
            if isinstance(img, str):
                # File path -- use directly, qwen_vl_utils handles paths
                image_blocks.append({"type": "image", "image": img})
            elif isinstance(img, Image.Image):
                # PIL Image -- convert to base64 data URI
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_uri = f"data:image/png;base64,{b64_str}"
                image_blocks.append({"type": "image", "image": data_uri})
            else:
                print(f"[QwenVL] Warning: skipping unsupported image type {type(img)}")

        return image_blocks

    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
        """
        Generate a claim draft using Qwen2-VL with interleaved image + text input.

        Args:
            prompt_vars: Dict with claimant/incident/vehicle/insurance info.
            images: Single image or list of images (PIL Image or file path strings).
                    These should be the segmented damage images.
            masks: Unused for now (segmentation already baked into images).

        Returns:
            str: Generated claim report text.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        # Build the text prompt
        prompt_text = self._build_prompt_text(prompt_vars)

        # Build multimodal message content
        content = []

        # Add image blocks first so the model "sees" the damage before reading the prompt
        image_blocks = self._prepare_image_content(images)
        content.extend(image_blocks)

        # Add the text prompt
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process images for the model
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate with conservative settings for structured output
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,        # greedy for consistency
                temperature=None,
                repetition_penalty=1.1, # reduce repetition in long outputs
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text


def get_llm_client(provider: str, api_key: str = None, model_name: str = None):
    if provider == "qwen-vl":
        return QwenVLLLMClient(model_name)
    elif provider == "gemini":
        return GeminiLLMClient(api_key, model_name)
    elif provider == "openai":
        return OpenAILLMClient(api_key, model_name)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")