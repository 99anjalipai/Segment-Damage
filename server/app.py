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

    uploaded_file = st.file_uploader("Upload Vehicle Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    event_description = st.text_area("Describe the event (e.g., 'I was rear-ended at a stop light...')", height=100)
    
    if uploaded_file is not None and event_description:
        try:
            image = Image.open(uploaded_file)
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)

            if st.button("🔍 Analyze Damage & Generate Claim"):
                api_key = st.secrets.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("⚠️ Please add your GOOGLE_API_KEY to server/.streamlit/secrets.toml to run the analysis.")
                    return
                
                with st.spinner("1️⃣ Running segmentation model to detect damage..."):
                    time.sleep(1.5)
                    img_array = np.array(image)
                    mask = segment_damage(img_array)
                    overlayed_image = overlay_mask(image, mask, color=(255, 50, 50), alpha=0.6)
                
                with col_img2:
                    st.markdown("#### Detected Damage")
                    st.image(overlayed_image, use_container_width=True, caption="Damage highlighted in red.")

                st.success("Damage successfully analyzed.")

                st.divider()
                st.markdown("### 📝 AI Claim Assistant")
                
                with st.spinner("2️⃣ Drafting insurance claim using Generative AI..."):
                    detected_damage = "a large dent and paint scratches on the primary body panel"
                    try:
                        drafter = ClaimDraftCore(api_key=api_key)
                        claim_draft = drafter.generate_draft(
                            detected_damage=detected_damage,
                            event_description=event_description
                        )
                        st.markdown('<div class="claim-box">', unsafe_allow_html=True)
                        st.markdown(claim_draft)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating claim draft: {e}")
        except Exception as e:
            st.error(f"Error processing inputs: {e}")

if __name__ == "__main__":
    main()
