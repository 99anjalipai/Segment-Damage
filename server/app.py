import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import os

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

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

def generate_mock_mask(image_shape):
    """
    Placeholder for the actual PyTorch Semantic Segmentation Model inference.
    Replace this with your actual model from models/segmentor.py
    """
    # Create a dummy mask (e.g., a circle representing a dent)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    center = (image_shape[1] // 2, image_shape[0] // 2)
    radius = min(image_shape[0], image_shape[1]) // 4
    cv2.circle(mask, center, radius, 1, -1)
    return mask

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    """
    Overlays a binary mask onto an image.
    """
    img_array = np.array(image).copy()
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    overlay = img_array.copy()
    overlay[mask == 1] = color
    
    output = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)
    return output

def draft_claim(api_key, detected_damage):
    """
    Uses LangChain to draft an insurance claim and provide advice.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    
    llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-1.5-pro")
    
    prompt = PromptTemplate(
        input_variables=["damage"],
        template="""
        You are an expert Auto Insurance Claims Adjuster and Advisor. 
        A user has uploaded an image of their vehicle, and our computer vision model has detected the following damage: {damage}.
        
        Please provide a response structured as follows:
        
        ### 1. Insurance Claim Draft
        Write a professional and concise formal letter/draft that the user can submit to their insurance company to report this incident and initiate the claim. Include placeholders like [Your Name], [Policy Number], etc.
        
        ### 2. Immediate Next Steps
        List 3-5 crucial steps the user should take immediately (e.g., taking more photos, reporting to police if necessary, not admitting fault).
        
        ### 3. Claim Tips
        Provide helpful insider tips for dealing with insurance adjusters and ensuring they get a fair payout for the detected damage ({damage}).
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({"damage": detected_damage})
    return response.content

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3204/3204066.png", width=60)
        st.title("AutoClaim AI")
        st.markdown("Automated Damage Assessment & Claim Generation.")
        
        
        st.divider()
        st.markdown("### About")
        st.info("This tool overlays computer vision segmentation masks over vehicle damage and uses LLMs to assist users in filing insurance claims.")

    # Main Content
    st.markdown('<p class="main-header">Vehicle Damage Assessment & Claims Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a photo of your damaged vehicle. Our AI will analyze the damage and draft a professional insurance claim for you.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Vehicle Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)

            if st.button("🔍 Analyze Damage & Generate Claim"):
                if "GOOGLE_API_KEY" not in st.secrets:
                    st.error("⚠️ Please add your GOOGLE_API_KEY to server/.streamlit/secrets.toml to run the analysis.")
                    return
                
                with st.spinner("1️⃣ Running segmentation model to detect damage..."):
                    # Simulate model inference time
                    time.sleep(1.5)
                    
                    # Convert PIL image to numpy array for processing
                    img_array = np.array(image)
                    
                    # TODO: Replace generate_mock_mask with actual Segmentor inference
                    # e.g., mask = segmentor.predict(img_array)
                    mask = generate_mock_mask(img_array.shape)
                    
                    # Overlay mask
                    overlayed_image = overlay_mask(image, mask, color=(255, 50, 50), alpha=0.6)
                
                with col2:
                    st.markdown("#### Detected Damage")
                    st.image(overlayed_image, use_container_width=True, caption="Damage highlighted in red.")

                st.success("Damage successfully analyzed: Identified significant dents and structural deformation.")

                st.divider()
                st.markdown("### 📝 AI Claim Assistant")
                
                with st.spinner("2️⃣ Drafting insurance claim and tips via LangChain..."):
                    # Mock detected damage string based on what the segmentor might output
                    detected_damage = "a large dent and paint scratches on the primary body panel"
                    
                    try:
                        claim_draft = draft_claim(st.secrets["GOOGLE_API_KEY"], detected_damage)
                        
                        st.markdown('<div class="claim-box">', unsafe_allow_html=True)
                        st.markdown(claim_draft)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating claim draft: {e}")
                        
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
