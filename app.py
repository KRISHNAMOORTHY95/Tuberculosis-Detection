import io
import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="TB X-ray Classifier",
    page_icon="🫁",
    layout="centered",
)

# Optional: make GPU memory growth safe if running on GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# ---------------------------
# App Constants
# ---------------------------
MODEL_PATH = "tb_classifier_resnet50.keras"
IMG_SIZE = 224
CATEGORIES = ['Normal', 'Tuberculosis']  # Must match training label order

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """Load a compiled Keras model from disk."""
    model = tf.keras.models.load_model(path)
    return model

def preprocess_image_pil(pil_img: Image.Image, target_size: int = 224) -> np.ndarray:
    """Resize -> to array -> scale to [0,1] -> add batch dim."""
    img = pil_img.convert("RGB").resize((target_size, target_size))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_softmax(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """Returns softmax probabilities regardless of model's final activation."""
    logits = model.predict(batch, verbose=0)
    if logits.ndim == 1:
        logits = np.expand_dims(logits, 0)
    row_sums = np.sum(logits, axis=1, keepdims=True)
    if np.allclose(row_sums, 1.0, atol=1e-3) and np.all(logits >= 0):
        probs = logits
    else:
        probs = tf.nn.softmax(logits, axis=1).numpy()
    return probs

def gradcam_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, last_conv_name: str = "conv5_block3_out"):
    """Compute Grad-CAM for ResNet50-style models."""
    try:
        last_conv_layer = model.get_layer(last_conv_name)
    except ValueError:
        return None
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    heatmap = tf.image.resize(heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    return heatmap

def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay a grayscale heatmap onto the original RGB image."""
    if heatmap is None:
        return None
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_uint8, mode="L").resize(pil_img.size)
    heatmap_img = heatmap_img.convert("RGBA")
    base = pil_img.convert("RGBA")
    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended.convert("RGB")

# ---------------------------
# Cover Image Helper (Online Default)
# ---------------------------
def show_cover_image():
    cover_image = None
    # Check for available local cover images
    for img_name in ["images.jpeg", "can-x-ray-detect-tuberculosis.jpg", "tuberculosis.jpg"]:
        if os.path.exists(img_name):
            cover_image = img_name
            break

    if cover_image:
        st.image(cover_image, caption="Tuberculosis Detection from X-rays", use_container_width=True)
    else:
        # Use online default image
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/8/8c/Chest_X-ray_PA_1.jpg",
            caption="Tuberculosis Detection from X-rays (Default Image)",
            use_container_width=True
        )

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigator")
choice = st.sidebar.selectbox('Go to', ['Introduction', 'TB X-Ray Prediction', 'About Me'])

# ---------------------------
# Pages
# ---------------------------
if choice == 'Introduction':
    st.title("Tuberculosis Detection from Chest X-rays")
    show_cover_image()

    st.subheader(
        'This system preprocesses and augments image data, applies deep learning models, '
        'and provides an interface to upload chest X-ray images and receive predictions.'
    )
    st.markdown(
        "- ✅ Transfer learning (ResNet50)\n"
        "- ✅ Clean UI with class probabilities\n"
        "- ✅ Optional Grad-CAM heatmap for explainability"
    )

elif choice == 'TB X-Ray Prediction':
    st.title("🫁 TB X-Ray Image Classification")
    st.write("Upload a chest X-ray image to classify it as **Normal** or **Tuberculosis**.")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Place it next to `app.py`.")
        st.stop()

    with st.spinner("Loading model..."):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    show_heatmap = st.checkbox("Show Grad-CAM heatmap (experimental)", value=False)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Could not read image: {e}")
            st.stop()

        st.image(image, caption='Uploaded Image', use_container_width=True)
        batch = preprocess_image_pil(image, target_size=IMG_SIZE)
        start = time.time()
        probs = predict_with_softmax(model, batch)
        infer_time = (time.time() - start) * 1000.0
        class_index = int(np.argmax(probs[0]))
        class_name = CATEGORIES[class_index]
        confidence = float(np.max(probs[0])) * 100.0
        st.success(f"**Prediction:** {class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.caption(f"Inference time: {infer_time:.1f} ms")
        st.write("Class probabilities:")
        prob_table = {CATEGORIES[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(CATEGORIES))}
        st.table(prob_table)

        if show_heatmap:
            with st.spinner("Generating Grad-CAM..."):
                heatmap = gradcam_heatmap(model, batch, last_conv_name="conv5_block3_out")
                if heatmap is None:
                    st.warning("Grad-CAM unavailable (layer not found or gradients not traceable).")
                else:
                    overlay = overlay_heatmap_on_image(image.resize((IMG_SIZE, IMG_SIZE)), heatmap, alpha=0.35)
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

elif choice == 'About Me':
    st.title('👩‍💻 Creator Info')
    if os.path.exists('AboutMe.webp'):
        st.image('AboutMe.webp', width=220)
    else:
        st.warning("Image `AboutMe.webp` not found.")
    st.markdown("""
**Developed by:** Krishnamoorthy K  
**Email:** mkrish818@gmail.com  

**Skills:** Computer Vision, Deep Learning, Python  

I’m passionate about learning fast and building practical AI applications!
""")
