import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Set model directory
MODEL_DIR = "models"
MODEL_FILES = {
    "ResNet50": "/content/drive/MyDrive/ResNet50_best (1).h5",
    "VGG16": "/content/drive/MyDrive/VGG16_best.h5",
    "EfficientNetB0": "/content/drive/MyDrive/EfficientNetB0_best.h5"
}

@st.cache_resource

def load_selected_model(model_path):
    return load_model(model_path)

def predict_tuberculosis(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction[0][0] > 0.5

def main():
    st.title("🪁 Tuberculosis Detection from Chest X-rays")

    # Select model
    selected_model_name = st.selectbox("Choose a model for prediction:", list(MODEL_FILES.keys()))
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model_name])

    # Load model
    if os.path.exists(model_path):
        model = load_selected_model(model_path)
    else:
        st.error(f"Model file not found: {model_path}")
        return

    # Image uploader
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                result = predict_tuberculosis(img, model)
                if result:
                    st.error("Prediction: Tuberculosis Detected ❌")
                else:
                    st.success("Prediction: Normal ✅")

if __name__ == '__main__':
    main()
