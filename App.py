import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Title
st.set_page_config(page_title="Tuberculosis Detection", layout="centered")
st.title("🩺 Tuberculosis Detection Using ResNet50")
st.markdown("Upload a chest X-ray image to predict if the person has tuberculosis.")

# Load model with caching
@st.cache_resource
def load_trained_model():
    model_path = "resnet50_best.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found! Please upload 'resnet50_best.h5' to your project files.")
        st.stop()
    return load_model(model_path)

model = load_trained_model()

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    result = "🟢 Normal" if prediction[0][0] < 0.5 else "🔴 Tuberculosis Detected"

    st.subheader("Prediction Result")
    st.success(result)
    st.write(f"Model Confidence: **{float(prediction[0][0]):.4f}**")

st.markdown("---")
st.caption("Model: ResNet50 | Developed by: YourName | Deployment: Streamlit Cloud")
