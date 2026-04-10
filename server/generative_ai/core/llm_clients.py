import os
import re
import time

from PIL import Image
import torch

from generative_ai.prompts.claim_prompts import CLAIM_DRAFT_PROMPT_TEMPLATE


DEFAULT_LOCAL_QWEN_MODEL = os.getenv("LOCAL_QWEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_LOCAL_QWEN_VL_MODEL = os.getenv("LOCAL_QWEN_VL_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
LOCAL_QWEN_VL_DEVICE = os.getenv("LOCAL_QWEN_VL_DEVICE", "auto").lower()
LOCAL_QWEN_VL_MAX_IMAGES = int(os.getenv("LOCAL_QWEN_VL_MAX_IMAGES", "1"))
LOCAL_QWEN_VL_MAX_EDGE = int(os.getenv("LOCAL_QWEN_VL_MAX_EDGE", "448"))
LOCAL_QWEN_DEBUG = os.getenv("LOCAL_QWEN_DEBUG", "1").lower() in {"1", "true", "yes", "on"}


REQUIRED_CLAIM_SECTION_HEADINGS = [
    "### 1. Insurance Claim Draft Letter",
    "### 2. Per-Image Damage Analysis",
    "### 3. Coverage and Policy Interpretation",
    "### 4. Immediate Next Steps (Next 24-72 Hours)",
]


CLAIM_SECTION_TITLES = {
    1: "Insurance Claim Draft Letter",
    2: "Per-Image Damage Analysis",
    3: "Coverage and Policy Interpretation",
    4: "Immediate Next Steps (Next 24-72 Hours)",
}


INVALID_INSURER_RESPONSE_PATTERNS = [
    r"we are writing to inform you",
    r"your claim has been processed",
    r"your claim has been accepted",
    r"the claim has already been opened",
    r"we have received all necessary documentation",
    r"thank you for choosing",
    r"we will be contacting you shortly",
    r"progressive insurance company",
]


def _get_preferred_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_preferred_torch_dtype(device: str):
    # Qwen-VL on MPS is prone to punctuation-collapse with float16; keep float32 there.
    if device == "mps":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def _get_preferred_vl_device() -> str:
    if LOCAL_QWEN_VL_DEVICE in {"cpu", "mps", "cuda"}:
        return LOCAL_QWEN_VL_DEVICE

    if torch.cuda.is_available():
        return "cuda"

    # Default to CPU for VL on Apple to avoid MPS aborts with large temporary NDArrays.
    if torch.backends.mps.is_available():
        return "cpu"

    return "cpu"


def _debug_log(message: str):
    if LOCAL_QWEN_DEBUG:
        print(f"[QWEN-DEBUG] {message}")


def _normalize_prompt_vars(prompt_vars: dict) -> dict:
    normalized = {}
    for key in CLAIM_DRAFT_PROMPT_TEMPLATE.input_variables:
        value = prompt_vars.get(key)
        if value is None:
            normalized[key] = "[PLACEHOLDER]"
        elif isinstance(value, str):
            stripped = value.strip()
            normalized[key] = stripped if stripped else "[PLACEHOLDER]"
        else:
            normalized[key] = value
    return normalized


def _is_complete_claim_draft(text: str) -> bool:
    if not text:
        return False

    return all(heading in text for heading in REQUIRED_CLAIM_SECTION_HEADINGS)


def _extract_claim_sections(text: str) -> dict[int, str]:
    if not text:
        return {}

    matches = list(
        re.finditer(r"(?m)^(?:#{1,6}\s*)?([1-4])\.\s+(.+?)\s*$", text)
    )
    if not matches:
        return {}

    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        section_number = int(match.group(1))
        if section_number not in CLAIM_SECTION_TITLES:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            previous_body = sections.get(section_number, "")
            if len(body) > len(previous_body):
                sections[section_number] = body

    return sections


def _claim_section_count(text: str) -> int:
    return len(_extract_claim_sections(text))


def _assemble_claim_draft_from_sections(sections: dict[int, str]) -> str:
    assembled_sections = []
    for section_number, heading in enumerate(REQUIRED_CLAIM_SECTION_HEADINGS, start=1):
        body = sections.get(section_number)
        if not body:
            continue
        assembled_sections.append(f"{heading}\n{body.strip()}")
    return "\n\n".join(assembled_sections).strip()


def _is_better_claim_candidate(candidate: str, current_best: str) -> bool:
    candidate_section_count = _claim_section_count(candidate)
    current_section_count = _claim_section_count(current_best)
    if candidate_section_count != current_section_count:
        return candidate_section_count > current_section_count
    return len(candidate.strip()) > len(current_best.strip())


def _looks_garbled_output(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True

    # Repeated punctuation collapse like "!!!!!!!!".
    if re.search(r"([!?.#*\-_=~])\1{24,}", cleaned):
        return True

    alpha_count = sum(1 for char in cleaned if char.isalpha())
    punct_count = sum(1 for char in cleaned if char in "!?.#*,-_=~")
    total_count = len(cleaned)
    if total_count == 0:
        return True

    alpha_ratio = alpha_count / total_count
    punct_ratio = punct_count / total_count
    return alpha_ratio < 0.2 and punct_ratio > 0.35


def _looks_like_insurer_response(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return True

    return any(re.search(pattern, cleaned) for pattern in INVALID_INSURER_RESPONSE_PATTERNS)


def _is_valid_claim_draft(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _looks_garbled_output(cleaned):
        return False
    if _looks_like_insurer_response(cleaned):
        return False
    return _is_complete_claim_draft(cleaned)


def _repair_claim_draft(prompt: str, draft_text: str, generator) -> str:
    _debug_log(
        f"repair.start chars={len((draft_text or '').strip())} sections={_claim_section_count(draft_text or '')}"
    )
    best_text = draft_text.strip()
    if _is_valid_claim_draft(best_text):
        normalized_sections = _extract_claim_sections(best_text)
        return _assemble_claim_draft_from_sections(normalized_sections) or best_text

    full_retry_prompt = (
        f"{prompt}\n\n"
        "Return the full claim with exactly these markdown headings and in this order:\n"
        "### 1. Insurance Claim Draft Letter\n"
        "### 2. Per-Image Damage Analysis\n"
        "### 3. Coverage and Policy Interpretation\n"
        "### 4. Immediate Next Steps (Next 24-72 Hours)\n"
        "Do not stop after Section 1. Do not omit any section.\n"
        "Write only from the claimant's perspective to the insurer.\n"
        "Do not write as the insurer and do not say the claim was already processed or accepted."
    )
    retry_text = generator(full_retry_prompt, 1200).strip()
    _debug_log(
        f"repair.retry chars={len(retry_text)} sections={_claim_section_count(retry_text)}"
    )
    if _is_better_claim_candidate(retry_text, best_text):
        best_text = retry_text

    merged_sections = {}
    for text in [draft_text, retry_text, best_text]:
        for section_number, body in _extract_claim_sections(text).items():
            if len(body) > len(merged_sections.get(section_number, "")):
                merged_sections[section_number] = body

    assembled_draft = _assemble_claim_draft_from_sections(merged_sections)
    if _is_valid_claim_draft(assembled_draft):
        _debug_log(
            f"repair.assembled chars={len(assembled_draft)} sections={_claim_section_count(assembled_draft)}"
        )
        return assembled_draft

    if _is_valid_claim_draft(best_text):
        _debug_log(f"repair.best-valid chars={len(best_text)} sections={_claim_section_count(best_text)}")
        return best_text

    _debug_log(f"repair.best chars={len(best_text)} sections={_claim_section_count(best_text)}")
    raise RuntimeError("Local Qwen produced an invalid claim draft perspective or incomplete sections.")


def _to_rgb_pil_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _mask_to_overlay_image(image, mask):
    base_image = _to_rgb_pil_image(image)
    if base_image is None or mask is None:
        return None

    overlay = base_image.copy()
    overlay_pixels = overlay.load()
    width, height = overlay.size

    if hasattr(mask, "shape"):
        mask_height, mask_width = mask.shape[:2]
    else:
        return None

    if (mask_width, mask_height) != (width, height):
        mask = Image.fromarray(mask).resize((width, height), resample=Image.NEAREST)
        mask = torch.from_numpy(__import__("numpy").array(mask))

    for y in range(height):
        for x in range(width):
            if int(mask[y][x]) > 0:
                red, green, blue = overlay_pixels[x, y]
                overlay_pixels[x, y] = (
                    min(255, int(red * 0.4 + 255 * 0.6)),
                    int(green * 0.4),
                    int(blue * 0.4),
                )

    return overlay


def _resize_for_vl(image: Image.Image, max_edge: int = LOCAL_QWEN_VL_MAX_EDGE) -> Image.Image:
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_edge:
        return image

    scale = max_edge / float(longest_edge)
    resized_width = max(28, int(width * scale))
    resized_height = max(28, int(height * scale))
    return image.resize((resized_width, resized_height), resample=Image.BILINEAR)


class LocalQwenChatClient:
    def __init__(self, model_name=None, max_new_tokens=400):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        requested_model = model_name or DEFAULT_LOCAL_QWEN_MODEL
        fallback_models = [
            requested_model,
            DEFAULT_LOCAL_QWEN_MODEL,
            "Qwen/Qwen1.5-1.8B-Chat",
            "Qwen/Qwen1.5-0.5B-Chat",
        ]
        self.max_new_tokens = max_new_tokens
        self.device = _get_preferred_torch_device()
        self.torch_dtype = _get_preferred_torch_dtype(self.device)
        load_errors = []

        for candidate in dict.fromkeys(fallback_models):
            try:
                _debug_log(
                    f"text.load.try model={candidate} device={self.device} dtype={self.torch_dtype}"
                )
                tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    dtype=self.torch_dtype,
                )
                model.to(self.device)
                model.eval()

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                self.model_name = candidate
                self.tokenizer = tokenizer
                self.model = model
                _debug_log(f"text.load.ok model={self.model_name}")
                break
            except Exception as exc:
                load_errors.append(f"{candidate}: {exc}")
                _debug_log(f"text.load.fail model={candidate} error={exc}")
        else:
            formatted_errors = " | ".join(load_errors)
            raise RuntimeError(f"Unable to load a local Qwen model. {formatted_errors}")

    def _build_chat_prompt(self, messages) -> str:
        formatted_messages = []
        for message in messages:
            if isinstance(message, str):
                formatted_messages.append({"role": "user", "content": message})
                continue

            role = getattr(message, "type", None)
            content = getattr(message, "content", None)
            if role == "human":
                formatted_messages.append({"role": "user", "content": content})
            elif role == "ai":
                formatted_messages.append({"role": "assistant", "content": content})
            elif role == "system":
                formatted_messages.append({"role": "system", "content": content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        transcript = []
        for message in formatted_messages:
            transcript.append(f"{message['role'].title()}: {message['content']}")
        transcript.append("Assistant:")
        return "\n".join(transcript)

    def _generate(self, prompt: str, max_new_tokens=None) -> str:
        started_at = time.time()
        token_budget = max_new_tokens or self.max_new_tokens
        _debug_log(
            f"text.generate.start model={self.model_name} tokens={token_budget} prompt_chars={len(prompt)}"
        )
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        tokenized = {key: value.to(self.model.device) for key, value in tokenized.items()}
        generation_kwargs = {
            "max_new_tokens": token_budget,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**tokenized, **generation_kwargs)

        generated_ids = output_ids[0][tokenized["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if text:
            _debug_log(f"text.generate.ok chars={len(text)} elapsed={time.time() - started_at:.2f}s")
            return text

        retry_prompt = f"{prompt}\nProvide a complete answer with all requested sections."
        retry_tokenized = self.tokenizer(retry_prompt, return_tensors="pt")
        retry_tokenized = {key: value.to(self.model.device) for key, value in retry_tokenized.items()}
        with torch.no_grad():
            retry_output_ids = self.model.generate(
                **retry_tokenized,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                min_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        retry_generated_ids = retry_output_ids[0][retry_tokenized["input_ids"].shape[1]:]
        retry_text = self.tokenizer.decode(retry_generated_ids, skip_special_tokens=True).strip()
        _debug_log(f"text.generate.retry chars={len(retry_text)} elapsed={time.time() - started_at:.2f}s")
        return retry_text

    def invoke(self, messages) -> str:
        prompt = self._build_chat_prompt(messages)
        return self._generate(prompt)

    def generate_from_prompt(self, prompt: str, max_new_tokens=1000) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._generate(prompt, max_new_tokens=max_new_tokens)


class LocalQwenVLClient:
    def __init__(self, model_name=None, max_new_tokens=1600):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

        requested_model = model_name or DEFAULT_LOCAL_QWEN_VL_MODEL
        fallback_models = [
            requested_model,
            DEFAULT_LOCAL_QWEN_VL_MODEL,
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
        ]
        self.max_new_tokens = max_new_tokens
        self.device = _get_preferred_vl_device()
        self.torch_dtype = _get_preferred_torch_dtype(self.device)
        load_errors = []

        for candidate in dict.fromkeys(fallback_models):
            try:
                _debug_log(
                    f"vl.load.try model={candidate} device={self.device} dtype={self.torch_dtype} use_fast=False"
                )
                processor = AutoProcessor.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    use_fast=False,
                    min_pixels=128 * 28 * 28,
                    max_pixels=384 * 28 * 28,
                )
                if "Qwen2.5-VL" in candidate:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        candidate,
                        trust_remote_code=True,
                        dtype=self.torch_dtype,
                    )
                else:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        candidate,
                        trust_remote_code=True,
                        dtype=self.torch_dtype,
                    )
                model.to(self.device)
                model.eval()

                # Avoid noisy warnings from incompatible sampling params baked into configs.
                if hasattr(model, "generation_config") and model.generation_config is not None:
                    model.generation_config.do_sample = False
                    model.generation_config.temperature = None
                    model.generation_config.top_p = None
                    model.generation_config.top_k = None

                self.model_name = candidate
                self.processor = processor
                self.model = model
                _debug_log(
                    f"vl.load.ok model={self.model_name} max_images={LOCAL_QWEN_VL_MAX_IMAGES} max_edge={LOCAL_QWEN_VL_MAX_EDGE}"
                )
                break
            except Exception as exc:
                load_errors.append(f"{candidate}: {exc}")
                _debug_log(f"vl.load.fail model={candidate} error={exc}")
        else:
            formatted_errors = " | ".join(load_errors)
            raise RuntimeError(f"Unable to load a local Qwen-VL model. {formatted_errors}")

    def generate_from_prompt(self, prompt: str, images=None, masks=None, max_new_tokens=None) -> str:
        started_at = time.time()
        visual_content = []
        processor_images = []

        selected_images = list(images or [])[:LOCAL_QWEN_VL_MAX_IMAGES]
        _debug_log(
            "vl.generate.start "
            f"model={self.model_name} tokens={max_new_tokens or self.max_new_tokens} "
            f"images={len(images or [])} selected={len(selected_images)} masks={len(masks or [])} "
            f"prompt_chars={len(prompt)}"
        )

        for index, image in enumerate(selected_images, start=1):
            pil_image = _to_rgb_pil_image(image)
            if pil_image is None:
                continue
            pil_image = _resize_for_vl(pil_image)

            # Prefer segmentation overlays when available; otherwise use the original image.
            selected_image = pil_image
            selected_label = f"Image {index}: original vehicle photo."

            if masks and len(masks) >= index:
                overlay_image = _mask_to_overlay_image(pil_image, masks[index - 1])
                if overlay_image is not None:
                    selected_image = _resize_for_vl(overlay_image)
                    selected_label = f"Image {index}: segmented overlay highlighting likely damaged regions."

            visual_content.append({"type": "image", "image": selected_image})
            visual_content.append({"type": "text", "text": selected_label})
            processor_images.append(selected_image)

        visual_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": visual_content}]

        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {
            "text": [chat_prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if processor_images:
            processor_kwargs["images"] = processor_images

        inputs = self.processor(**processor_kwargs)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        _debug_log(
            "vl.generate.inputs "
            f"input_ids={tuple(inputs['input_ids'].shape) if 'input_ids' in inputs else 'n/a'} "
            f"pixel_values={tuple(inputs['pixel_values'].shape) if 'pixel_values' in inputs else 'n/a'}"
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )

        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        outputs = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_text = outputs[0].strip() if outputs else ""
        _debug_log(
            "vl.generate.done "
            f"chars={len(output_text)} garbled={_looks_garbled_output(output_text)} elapsed={time.time() - started_at:.2f}s"
        )
        return output_text

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


class QwenLLMClient(BaseLLMClient):
    def __init__(self, model_name):
        self.client = LocalQwenChatClient(model_name=model_name, max_new_tokens=900)

    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
        _debug_log(f"claim.text.start images={len(images or [])} masks={len(masks or [])}")
        normalized_prompt_vars = _normalize_prompt_vars(prompt_vars)
        prompt = CLAIM_DRAFT_PROMPT_TEMPLATE.format(**normalized_prompt_vars)
        response_text = self.client.generate_from_prompt(prompt, max_new_tokens=900)
        return _repair_claim_draft(
            prompt,
            response_text,
            lambda next_prompt, token_budget: self.client.generate_from_prompt(next_prompt, max_new_tokens=token_budget),
        )

class QwenVLLLMClient(BaseLLMClient):
    def __init__(self, model_name):
        self.client = LocalQwenVLClient(model_name=model_name, max_new_tokens=480)
    
    def generate(self, prompt_vars: dict, images=None, masks=None) -> str:
        _debug_log(f"claim.vl.start images={len(images or [])} masks={len(masks or [])}")
        normalized_prompt_vars = _normalize_prompt_vars(prompt_vars)
        prompt = CLAIM_DRAFT_PROMPT_TEMPLATE.format(**normalized_prompt_vars)
        response_text = self.client.generate_from_prompt(prompt, images=images, masks=masks, max_new_tokens=480)

        if _looks_garbled_output(response_text):
            _debug_log("claim.vl.garbled initial=true; running recovery prompt")
            recovery_prompt = (
                f"{prompt}\n\n"
                "Your previous output was malformed. Use normal English sentences and markdown headings. "
                "Do not output repeated punctuation characters.\n"
                "Write from the claimant's perspective to the insurer, not as an insurer response."
            )
            response_text = self.client.generate_from_prompt(
                recovery_prompt,
                images=images,
                masks=masks,
                max_new_tokens=360,
            )

        if _looks_garbled_output(response_text) or _looks_like_insurer_response(response_text):
            _debug_log("claim.vl.invalid recovery=true; raising error")
            raise RuntimeError("Local Qwen-VL produced invalid claim-draft output on the current device.")

        return _repair_claim_draft(
            prompt,
            response_text,
            lambda next_prompt, token_budget: self.client.generate_from_prompt(
                next_prompt,
                images=images,
                masks=masks,
                max_new_tokens=token_budget,
            ),
        )

def get_llm_client(provider: str, api_key: str, model_name: str):
    if provider == "qwen-vl":
        return QwenVLLLMClient(model_name)
    if provider == "qwen":
        return QwenLLMClient(model_name)
    if provider == "gemini":
        return GeminiLLMClient(api_key, model_name)
    elif provider == "openai":
        return OpenAILLMClient(api_key, model_name)
    # Add more providers here
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
