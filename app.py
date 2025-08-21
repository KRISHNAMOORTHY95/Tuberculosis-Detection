import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load model once
@st.cache_resource
def load_tb_model():
    model = load_model("best_model.h5")
    return model

model = load_tb_model()

# Image preprocessing function
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.set_page_config(page_title="Tuberculosis Detection", layout="centered")
st.title("🫁 Tuberculosis Detection from Chest X-rays")
st.write("Upload a chest X-ray image and the model will predict whether TB is detected.")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and predict
    uploaded_file.seek(0)  # Reset pointer
    processed_img = preprocess_image(uploaded_file)
    prediction = model.predict(processed_img)

    # Assuming binary classification (TB vs Normal)
    prob = prediction[0][0]
    label = "Tuberculosis Detected" if prob > 0.5 else "Normal"

    st.subheader(f"Prediction: {label}")
    st.progress(float(prob) if prob <= 1 else 1)
    st.write(f"**Confidence:** {prob*100:.2f}%")

st.markdown("---")
st.caption("Powered by Deep Learning • Built with Streamlit")
