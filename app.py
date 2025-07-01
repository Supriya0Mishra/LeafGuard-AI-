import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Setup
st.set_page_config(page_title="LeafGuard-AI 🌿", layout="centered")

# 🔥 Custom CSS for Modern UI
st.markdown("""
    <style>
    body {
        background-color: #f0f4f7;
        font-family: 'Segoe UI', sans-serif;
    }

    .main {
        background: linear-gradient(145deg, #ffffff, #f4f4f4);
        border-radius: 15px;
        padding: 40px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2e7d32;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }

    .upload-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #66bb6a;
        margin-bottom: 20px;
    }

    .prediction-box {
        background-color: #ffffff;
        border-left: 5px solid #66bb6a;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .footer {
        font-size: 13px;
        color: #888;
        text-align: center;
        margin-top: 50px;
    }

    .stButton>button {
        background-color: #388e3c;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        padding: 10px 25px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🌿 LeafGuard-AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Disease Detection for Mulberry Leaves </div>", unsafe_allow_html=True)

# Class Labels
class_names_model1 = ['Healthy', 'Bacterial Blight', 'Powdery Mildew', 'Leaf Spot', 'Rust', 'Other']
class_names_model2 = ['Bacterial Blight', 'Healthy', 'Leaf Spot', 'Powdery Mildew', 'Rust']

# Load Models
model1 = tf.keras.models.load_model(
    r"C:\Users\Supriya Mishra\OneDrive\Desktop\LeafGuard-AI\LeafGuard-AI\LeafGuard-AI\model\leaf_disease_model1.h5"
)
model2 = tf.keras.models.load_model(
    r"C:\Users\Supriya Mishra\OneDrive\Desktop\LeafGuard-AI\LeafGuard-AI\LeafGuard-AI\model\leaf_disease_model2.h5"
)

input_size_model1 = model1.input_shape[1:3]
input_size_model2 = model2.input_shape[1:3]

def preprocess_image(image, size):
    image = image.resize(size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Upload Section
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📸 Upload a Mulberry Leaf Image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    try:
        img_tensor1 = preprocess_image(image, input_size_model1)
        img_tensor2 = preprocess_image(image, input_size_model2)

        with st.spinner("🔬 Running diagnosis using Model 1..."):
            prediction1 = model1.predict(img_tensor1)[0]

        with st.spinner("🧪 Running diagnosis using Model 2..."):
            prediction2 = model2.predict(img_tensor2)[0]

        st.subheader("🔍 Results")

        # Prediction from Model 1
        top_idx1 = np.argmax(prediction1)
        st.markdown(f"<div class='prediction-box'><strong>🧠 Model 1 Prediction:</strong> <code>{class_names_model1[top_idx1]}</code> ({prediction1[top_idx1]*100:.2f}%)</div>", unsafe_allow_html=True)
        st.bar_chart({name: float(pred) for name, pred in zip(class_names_model1, prediction1)})

        # Prediction from Model 2
        top_idx2 = np.argmax(prediction2)
        st.markdown(f"<div class='prediction-box'><strong>🔍 Model 2 Prediction:</strong> <code>{class_names_model2[top_idx2]}</code> ({prediction2[top_idx2]*100:.2f}%)</div>", unsafe_allow_html=True)
        st.bar_chart({name: float(pred) for name, pred in zip(class_names_model2, prediction2)})

    except Exception as e:
        st.error(f"🚫 An error occurred during prediction: {e}")

# Footer
st.markdown("<div class='footer'>Made with 💚 by Supriya Mishra | © 2025</div>", unsafe_allow_html=True)
