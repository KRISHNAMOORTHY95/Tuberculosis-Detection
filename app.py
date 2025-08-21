import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

choice=st.sidebar.selectbox('Navigator',['Introduction','TB X-Ray Prediction','About Me'])
if choice=='Introduction':
    st.title('Tuberculosis X-rays Classification')
    st.image('images.jpeg')
    st.subheader('The system will preprocess and augment image data, train multiple deep learning models, and evaluate their performance. The final application will provide an interface for uploading X-ray images and receiving predictions')

elif choice=='TB X-Ray Prediction':
    MODEL_PATH = 'tb_classifier_resnet50.keras'
    IMG_SIZE = 224
    CATEGORIES = ['Normal', 'Tuberculosis']

    # --- Load Model ---
    @st.cache_resource
    def load_model(path):
        """Loads the pre-trained Keras model."""
        try:
            model = tf.keras.models.load_model(path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_model(MODEL_PATH)

    # --- UI Elements ---
    st.title("🫁 TB X-Ray Image Classification")
    st.write("Upload a chest X-ray image to classify it as Normal or containing Tuberculosis.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        # --- Preprocess Image ---
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Resize and normalize
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        img_array /= 255.0

        # --- Make Prediction ---
        with st.spinner('Classifying...'):
            prediction = model.predict(img_array)
            score = tf.nn.softmax(prediction[0])
            class_index = np.argmax(score)
            class_name = CATEGORIES[class_index]
            confidence = 100 * np.max(score)

        # --- Display Result ---
        st.success(f"**Prediction:** {class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")


elif choice == 'About Me':
    st.title('👩‍💻 Creator Info')
    if os.path.exists('AboutMe.webp'):
        st.image('AboutMe.webp', width=220)
    else:
        st.warning("Image `AboutMe.webp` not found.")
    st.markdown("""
**Developed by:** Krishnamoorthy  
**Email:** mkrish818@gmail.com

**Skills:** Computer Vision, Deep Learning, Python

I’m passionate about learning fast and building practical AI applications!
""")




