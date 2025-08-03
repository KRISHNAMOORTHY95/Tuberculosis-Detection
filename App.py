import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set title
st.title("🩺 Tuberculosis Detection from Chest X-ray")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("best_model.h5")  # or .keras if that's your saved file

model = load_trained_model()

# Image upload
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

# Preprocess image
def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        input_image = preprocess(img)
        prediction = model.predict(input_image)[0][0]
        
        if prediction > 0.5:
            st.error("⚠️ Likely to have **Tuberculosis**")
        else:
            st.success("✅ Likely **Normal** Chest X-ray")

