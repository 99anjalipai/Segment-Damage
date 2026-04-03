# Generative AI Explorations

This module contains our generative AI pipelines for assessing insurance documents, user events, and segmentation masks (or text interpretations of them) to generate comprehensive insurance claim drafts.

## Structure
- `core/`: Core orchestrators and logic for calling the Models.
- `prompts/`: Standardized prompts for zero-shot, few-shot, and Chain of Thought.
- `services/`: VLM processing services (e.g. running Gemini with multimodal inputs).
- `utils/`: Helpers for document OCR, image conversion, and data formatting.

## Usage
1. Setup your API keys (`GOOGLE_API_KEY`).
2. Implement your VLM pipeline directly in the `core/` modules.
3. Import the required class to the Streamlit app.
