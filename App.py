import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

st.set_page_config(page_title="🩺 Tuberculosis Detection", layout="centered")
st.title("🩺 Tuberculosis Detection from Chest X-ray")

@st.cache_resource
def load_trained_model():
    try:
        return load_model("resnet50_best.h5")  # Make sure the model file is uploaded in the same directory
    except FileNotFoundError:
        st.error("Model file not found. Please upload 'resnet50_best.h5' to the app directory.")
        st.stop()

model = load_trained_model()

st.markdown("Upload a chest X-ray image and get a prediction whether the person may have Tuberculosis or not.")

uploaded_file = st.file_uploader("Choose a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        resized_image = image.resize((224, 224))
        img_array = np.expand_dims(np.array(resized_image) / 255.0, axis=0)

        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        if st.button("🧠 Predict TB"):
            prediction = model.predict(img_array)[0][0]
            label = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("---")
st.caption("Built with ❤ using Streamlit, TensorFlow and Transfer Learning")
