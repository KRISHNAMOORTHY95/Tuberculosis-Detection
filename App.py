import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Tuberculosis Detection", layout="centered")
st.title("🫁 Tuberculosis Detection using ResNet50")
st.write("Upload a Chest X-ray image and let the model predict whether TB is present.")

@st.cache_resource
def load_trained_model():
    model = load_model("/content/drive/MyDrive/ResNet50_best.h5")
    return model

model = load_trained_model()

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = preprocess_image(img)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)
        predicted_class = "Tuberculosis Detected" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

st.markdown("---")
st.markdown("Developed using ResNet50 and Streamlit 🧠")
