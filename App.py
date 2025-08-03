import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="🩺 Tuberculosis Detection from Chest X-ray", layout="centered")
st.title("🩺 Tuberculosis Detection from Chest X-ray")

@st.cache_resource
def load_trained_model():
    model = load_model("resnet50_best.h5")  # Make sure this file is uploaded to Streamlit Cloud
    return model

model = load_trained_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict TB"):  
        preprocessed_img = preprocess_image(img)
        prediction = model.predict(preprocessed_img)
        class_names = ['Normal', 'Tuberculosis']
        result = class_names[int(np.round(prediction[0][0]))]

        st.success(f"Prediction: **{result}**")

        if result == "Tuberculosis":
            st.error("⚠️ The model detected signs of Tuberculosis. Please consult a medical professional.")
        else:
            st.success("✅ The model did not detect signs of Tuberculosis.")

st.markdown("---")
st.caption("This app uses a ResNet50-based model trained on chest X-ray images.")
