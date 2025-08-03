import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"D:\DS projects\TB\tb_detection_model_.keras")

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Set page config
st.set_page_config(page_title="🩻 TB Detection App", page_icon="🧬", layout="centered")

# Custom header
st.markdown("<h1 style='text-align: center; color: teal;'>🩻 Tuberculosis Detection from Chest X-ray</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a chest X-ray image and let our AI model predict if there's any sign of TB.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded X-ray Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0][0]
        label = "Tuberculosis Detected but don't worry🛑" if prediction > 0.5 else "Its Normal,great ✅"
        confidence = prediction if prediction > 0.5 else 1 - prediction

    st.success(f"### 🔎 Prediction: **{label}**")
    st.info(f"📊 Confidence: `{confidence * 100:.2f}%`")

    # Add an expandable info box
    with st.expander("ℹ️ About the Model"):
        st.write("""
            - This model is trained to detect signs of Tuberculosis in chest X-ray images.
            - Input size: 224x224 pixels.
            - Output: Binary classification (TB or Normal).
            - Built with TensorFlow & Keras.
        """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by Uvagai</p>", unsafe_allow_html=True)
