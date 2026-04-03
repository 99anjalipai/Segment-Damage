import streamlit as st
import numpy as np
import time
from PIL import Image

# Import separated services
from services.segmentation import segment_damage, overlay_mask
from generative_ai.core.claim_drafter import ClaimDraftCore

st.set_page_config(
    page_title="AutoClaim AI - Vehicle Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .claim-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3204/3204066.png", width=60)
        st.title("AutoClaim AI")
        st.markdown("Automated Damage Assessment & Claim Generation.")
        
        st.divider()
        st.markdown("### About")
        st.info("This tool overlays computer vision segmentation masks over vehicle damage and uses Generative AI to assist users in filing insurance claims based on context and event descriptions.")

    # Main Content
    st.markdown('<p class="main-header">Vehicle Damage Assessment & Claims Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a photo of your damaged vehicle. Our AI will analyze the damage and draft a professional insurance claim for you using policy context and event details.</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📝 Standard Claim Form", "💬 Interactive Chat Assistant"])

    with tab1:
        # --- Model selection ---
        import os
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        model_options = []
        for folder in os.listdir(outputs_dir):
            folder_path = os.path.join(outputs_dir, folder)
            if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "best.pt")):
                model_options.append(folder)
        model_name = st.selectbox("Select Segmentation Model", model_options, index=0 if model_options else None)


        # --- Collect all required info ---
        st.markdown("### Claimant Information")
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Your Name")
            user_address = st.text_input("Your Address")
            user_phone = st.text_input("Your Phone Number")
            user_email = st.text_input("Your Email Address")
        with col2:
            insurance_company = st.text_input("Insurance Company Name")
            policy_number = st.text_input("Policy Number")
            incident_date = st.date_input("Date of Incident")
            incident_time = st.text_input("Time of Incident")
            incident_location = st.text_input("Location of Incident")

        st.markdown("### Vehicle Information")
        col3, col4 = st.columns(2)
        with col3:
            vehicle_year = st.text_input("Vehicle Year")
            vehicle_make = st.text_input("Vehicle Make")
            vehicle_model = st.text_input("Vehicle Model")
        with col4:
            vehicle_vin = st.text_input("Vehicle VIN")
            license_plate = st.text_input("License Plate Number")

        # --- Multiple image upload ---
        uploaded_files = st.file_uploader("Upload Vehicle Images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        event_description = st.text_area("Describe the event (e.g., 'I was rear-ended at a stop light...')", height=100)
        generate_claim = st.checkbox("Generate AI Claim Draft (optional, saves credits)", value=True)



        # --- Validation ---
        required_fields = [user_name, user_address, user_phone, user_email, insurance_company, policy_number, incident_date, incident_time, incident_location, vehicle_year, vehicle_make, vehicle_model, vehicle_vin, license_plate, event_description]
        all_fields_filled = all(str(x).strip() for x in required_fields)

        if uploaded_files and all_fields_filled:
            try:
                images = [Image.open(f) for f in uploaded_files]
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("#### Original Images")
                    for img in images:
                        st.image(img, use_container_width=True)

                if st.button("🔍 Analyze Damage" + (" & Generate Claim" if generate_claim else "")):
                    from services import segmentation as seg
                    masks = []
                    overlayed_images = []
                    with st.spinner("1️⃣ Running segmentation model to detect damage on all images..."):
                        time.sleep(1.5)
                        for idx, img in enumerate(images):
                            img_array = np.array(img)
                            mask = seg.segment_damage(img_array, model_name=model_name)
                            overlayed = seg.overlay_mask(img, mask, color=(255, 50, 50), alpha=0.6)
                            masks.append(mask)
                            overlayed_images.append(overlayed)

                    with col_img2:
                        st.markdown("#### Detected Damage")
                        for overlayed in overlayed_images:
                            st.image(overlayed, use_container_width=True, caption="Damage highlighted in red.")

                    st.success("Damage successfully analyzed.")

                    if generate_claim:
                        st.divider()
                        st.markdown("### 📝 AI Claim Assistant")
                        with st.spinner("2️⃣ Drafting insurance claim using Qwen..."):
                            # Compose all info for the claim generator
                            user_info = {
                                "user_name": user_name,
                                "user_address": user_address,
                                "user_phone": user_phone,
                                "user_email": user_email,
                            }
                            insurance_info = {
                                "insurance_company": insurance_company,
                                "policy_number": policy_number,
                            }
                            incident_info = {
                                "incident_date": str(incident_date),
                                "incident_time": incident_time,
                                "incident_location": incident_location,
                                "event_description": event_description,
                            }
                            vehicle_info = {
                                "vehicle_year": vehicle_year,
                                "vehicle_make": vehicle_make,
                                "vehicle_model": vehicle_model,
                                "vehicle_vin": vehicle_vin,
                                "license_plate": license_plate,
                            }
                            try:
                                drafter = ClaimDraftCore(provider="qwen-vl", model_name="Qwen/Qwen1.5-0.5B-Chat")
                                claim_draft = drafter.generate_draft(
                                    detected_damage="Detected damage on multiple images.",
                                    event_description=event_description,
                                    images=images,
                                    masks=masks,
                                    user_info=user_info,
                                    insurance_info=insurance_info,
                                    incident_info=incident_info,
                                    vehicle_info=vehicle_info
                                )
                                st.markdown('<div class="claim-box">', unsafe_allow_html=True)
                                st.markdown(claim_draft)
                                st.markdown('</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error generating claim draft: {e}")
            except Exception as e:
                st.error(f"Error processing inputs: {e}")
        elif uploaded_files and not all_fields_filled:
            st.warning("Please fill in all required fields above before generating a claim draft.")

    with tab2:
        st.markdown("### 💬 Interactive Chat Assistant")
        st.info("Chat with our AI to easily gather your claim details in a natural way. You can text or use the image uploader below.")
        
        # Adding image upload capabilities to the chat assistant
        chat_images = st.file_uploader("Upload Vehicle Images for the Chat Assistant (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="chat_uploader")
        
        if chat_images and "chat_images_processed" not in st.session_state:
            try:
                images_opened = [Image.open(f) for f in chat_images]
                from services import segmentation as seg
                masks = []
                overlayed_images = []
                with st.spinner("Analyzing uploaded images..."):
                    for img in images_opened:
                        img_array = np.array(img)
                        mask = seg.segment_damage(img_array, model_name=model_name)
                        overlayed = seg.overlay_mask(img, mask, color=(255, 50, 50), alpha=0.6)
                        masks.append(mask)
                        overlayed_images.append(overlayed)
                
                st.session_state.chat_images_processed = True
                st.session_state.chat_raw_images = images_opened
                st.session_state.chat_masks = masks
                st.session_state.chat_overlayed = overlayed_images
                # Automatically tell the LLM that images were added
                if "messages" in st.session_state:
                    from langchain_core.messages import HumanMessage
                    import base64
                    from io import BytesIO
                    st.session_state.messages.append(HumanMessage(content="(System: User has uploaded vehicle photos with segmented damage. Acknowledge this and ask for details on how the accident occurred.)"))
            except Exception as e:
                st.error(f"Error processing images: {e}")
        
        if "chat_overlayed" in st.session_state:
            with st.expander("View Analyzed Vehicle Images", expanded=False):
                col_a, col_b = st.columns(2)
                for idx, (raw, over) in enumerate(zip(st.session_state.chat_raw_images, st.session_state.chat_overlayed)):
                    with col_a:
                        st.image(raw, caption=f"Original {idx+1}")
                    with col_b:
                        st.image(over, caption=f"Analyzed {idx+1}")

        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.warning("⚠️ Please add your GOOGLE_API_KEY to server/.streamlit/secrets.toml to use the chat.")
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                if "chat_llm" not in st.session_state:
                    # Switch from Gemini to Qwen (HuggingFace pipeline wrapper) for chatbot 
                    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
                    from transformers import pipeline
                    import torch
                    
                    try:
                        # Load Qwen model via huggingface pipeline
                        pipe = pipeline(
                            "text-generation", 
                            model="Qwen/Qwen1.5-0.5B-Chat", # Using 0.5B as default for chatbot responsiveness
                            max_new_tokens=256,
                            device=torch.device("cpu")
                        )
                        st.session_state.chat_llm = HuggingFacePipeline(pipeline=pipe)
                        st.session_state.is_huggingface = True
                    except Exception as e:
                        st.error(f"Failed to load local Qwen model: {e}")
                        st.stop()
                
                if "messages" not in st.session_state:
                    st.session_state.messages = [
                        SystemMessage(content="You are an expert Auto Insurance Claims Assistant. Your goal is to gather the user's name, insurance company, policy number, vehicle details, and details about the incident in a friendly, conversational manner. Ask for one or two pieces of information at a time so you don't overwhelm them. Please ask the user to upload images of the damage if they haven't already. Once you have all the information, summarize it strictly and concisely and ask if they are ready to proceed with generating the official claim draft. Be polite and professional."),
                        AIMessage(content="Hello! I am your AI Claims Assistant. I'm here to help you file your auto insurance claim. To get started, could you please tell me your name and the name of your insurance provider, and upload some photos of the damage using the uploader above?")
                    ]
                    
                # Display chat messages (excluding SystemMessage)
                for msg in st.session_state.messages:
                    if isinstance(msg, AIMessage):
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(msg.content)
                    elif isinstance(msg, HumanMessage) and not msg.content.startswith("(System:"):
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(msg.content)
                            
                # Unified audio/text handling
                prompt = st.chat_input("Type your message here...")
                # Removed generic st.audio_input as it does not inherently transcribe speech to text.
                
                if prompt:
                    # Add user message
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(prompt)
                        
                    # Generate response
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Thinking..."):
                            # Since HF pipeline takes strings rather than Langchain Message objects directly
                            if st.session_state.get("is_huggingface"):
                                prompt_str = "\n".join([m.content for m in st.session_state.messages])
                                response_text = st.session_state.chat_llm.invoke(prompt_str)
                            else:
                                response = st.session_state.chat_llm.invoke(st.session_state.messages)
                                response_text = response.content
                                
                            st.markdown(response_text)
                            st.session_state.messages.append(AIMessage(content=response_text))
                            
                if "messages" in st.session_state and len(st.session_state.messages) > 4:
                    if st.button("📝 Generate Official Claim Draft with Qwen-VL (Free)"):
                        with st.spinner("Drafting insurance claim using Qwen-VL..."):
                            try:
                                # Prepare gathered chat context as event description
                                chat_context = "\n".join([m.content for m in st.session_state.messages if isinstance(m, HumanMessage) or isinstance(m, AIMessage)])
                                drafter = ClaimDraftCore(provider="qwen-vl", model_name="Qwen/Qwen-VL-Chat") # Using Qwen-VL
                                claim_draft = drafter.generate_draft(
                                    detected_damage="Detected damage on uploaded images.",
                                    event_description="Based on conversation:\n" + chat_context,
                                    images=st.session_state.get("chat_raw_images", []),
                                    masks=st.session_state.get("chat_masks", [])
                                )
                                st.markdown('<div class="claim-box">', unsafe_allow_html=True)
                                st.markdown(claim_draft)
                                st.markdown('</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error generating claim draft: {e}")

            except ImportError:
                st.error("Missing required LangChain or GenerativeAI packages. Please install them to use the chat.")

if __name__ == "__main__":
    main()
