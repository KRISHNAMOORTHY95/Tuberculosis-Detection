import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Set page title and layout
st.set_page_config(page_title="Tuberculosis Detection", layout="centered")
st.title("🩺 Tuberculosis Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect whether Tuberculosis is present.")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = "ResNet50_best.h5"  # Updated to match your best model filename
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    model = load_model(model_path)
    return model

model = load_trained_model()

# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Upload and predict
uploaded_file = st.file_uploader("📂 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing...")

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)

    class_names = ["Normal", "Tuberculosis"]
    predicted_class = class_names[int(prediction[0][0] > 0.5)]
    confidence = float(prediction[0][0]) if predicted_class == "Tuberculosis" else 1 - float(prediction[0][0])

    st.success(f"🧠 **Prediction**: {predicted_class}")
    st.info(f"📊 **Confidence**: {confidence * 100:.2f}%")
