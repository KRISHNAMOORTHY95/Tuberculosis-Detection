import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Paths
MODEL_DIR = 'models'
MODEL_OPTIONS = {
    'ResNet50': f'{MODEL_DIR}/ResNet50_best.h5',
    'VGG16': f'{MODEL_DIR}/VGG16_best.h5',
    'EfficientNetB0': f'{MODEL_DIR}/EfficientNetB0_best.h5',
}

IMG_SIZE = (224, 224)

st.set_page_config(page_title="TB Detection", layout="centered")
st.title("🪁 Tuberculosis Detection from Chest X-rays")

# Model selector
model_choice = st.selectbox("Choose a model for prediction:", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[model_choice]

# Load model once
@st.cache_resource
def load_selected_model(path):
    return load_model(path)

model = load_selected_model(model_path)

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded X-ray', use_column_width=True)

    if st.button("🔍 Predict"):
        img_resized = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Tuberculosis Detected 🚩" if prediction > 0.5 else "Normal ✅"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### 🔎 Prediction: `{label}`")
        st.progress(float(confidence))
        st.write(f"Confidence: `{confidence:.2f}`")
