import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Tuberculosis Detection", layout="centered")
st.title("🩺 Tuberculosis Detection from Chest X-Rays")
st.write("Upload a chest X-ray image and select the model to predict Tuberculosis.")

# Load model only once
@st.cache_resource
def load_trained_model():
    model_path = "ResNet50_best.h5"  # Uploaded model file name
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    model = load_model(model_path)
    return model

model = load_trained_model()

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]
    prediction_label = "Tuberculosis Detected" if prediction >= 0.5 else "Normal"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### 🧪 Prediction: **{prediction_label}**")
    st.markdown(f"#### Confidence Score: `{confidence:.2f}`")

    st.success("Prediction complete.")
else:
    st.info("Please upload a chest X-ray image to begin.")
