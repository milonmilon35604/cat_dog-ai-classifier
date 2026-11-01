import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="PetScan AI — Cat vs Dog",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1e293b 0%, #0f172a 90%);
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.2rem;
    }

    .badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .badge-dog {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid #38bdf8;
        color: #38bdf8;
    }
    .badge-cat {
        background: rgba(244, 114, 182, 0.15);
        border: 1px solid #f472b6;
        color: #f472b6;
    }
    .badge-human {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1, #3b82f6);
        color: #ffffff;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.5rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 40px;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown('<div class="main-title">🐾 Cat vs Dog AI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image and let AI analyze whether it is a Cat or a Dog.</div>', unsafe_allow_html=True)

# -----------------------
# Load Model & Face Detector
# -----------------------
@st.cache_resource
def load_model():
    model = tf.saved_model.load("save_model")
    return model

@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

model = load_model()
infer = model.signatures["serving_default"]
face_cascade = load_face_detector()

# -----------------------
# File Uploader
# -----------------------
uploaded_file = st.file_uploader(
    "📸 Upload Image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, and PNG formats"
)

# -----------------------
# Image Processing & Prediction
# -----------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    np_img = np.array(image)

    col_img, col_pred = st.columns([1.1, 1], gap="large")

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Picture", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Analysis Dashboard")

        if st.button("⚡ Run Classification"):
            with st.spinner("Analyzing image features..."):
                # 1. Human Face Check using OpenCV
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    st.markdown('<div class="badge badge-human">👤 HUMAN DETECTED</div>', unsafe_allow_html=True)
                    st.error("This appears to be a human picture, not a cat or dog.")
                else:
                    # 2. Run Classification Model
                    img = image.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                    prediction = infer(tf.constant(img_array))
                    raw_score = list(prediction.values())[0].numpy()[0][0]

                    # Standard classification boundary
                    if raw_score >= 0.50:
                        confidence = float(raw_score * 100)
                        st.markdown('<div class="badge badge-dog">🐶 DOG DETECTED</div>', unsafe_allow_html=True)
                        st.progress(min(int(confidence), 100) / 100.0)
                        st.metric(label="Certainty", value=f"{confidence:.2f}%")
                        st.balloons()
                    else:
                        confidence = float((1.0 - raw_score) * 100)
                        st.markdown('<div class="badge badge-cat">🐱 CAT DETECTED</div>', unsafe_allow_html=True)
                        st.progress(min(int(confidence), 100) / 100.0)
                        st.metric(label="Certainty", value=f"{confidence:.2f}%")
        else:
            st.info("Click **⚡ Run Classification** to identify the pet.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 40px; color: #64748b;">
            👆 Upload an image above to start the classification.
        </div>
    """, unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Made by <strong>Milon Ghosh</strong>
    </div>
    """,
    unsafe_allow_html=True
)