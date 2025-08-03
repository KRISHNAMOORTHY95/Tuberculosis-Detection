import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Title
st.title("Tuberculosis Detection using ResNet50")

# Load model from uploaded file
@st.cache_resource
def load_trained_model():
    model_path = "ResNet50_best.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return load_model(model_path)

model = load_trained_model()

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    if model:
        prediction = model.predict(img_array)[0][0]
        label = "Tuberculosis Detected" if prediction >= 0.5 else "Normal"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.subheader("Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.warning("Model not loaded. Please ensure the model file is uploaded.")
