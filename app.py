import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

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
    """
    Returns softmax probabilities regardless of model's final activation.
    If your model already ends with softmax, this is still fine.
    """
    logits = model.predict(batch, verbose=0)
    # Ensure 2D
    if logits.ndim == 1:
        logits = np.expand_dims(logits, 0)
    # If already sums ~1, treat as probs; else softmax
    row_sums = np.sum(logits, axis=1, keepdims=True)
    if np.allclose(row_sums, 1.0, atol=1e-3) and np.all(logits >= 0):
        probs = logits
    else:
        probs = tf.nn.softmax(logits, axis=1).numpy()
    return probs

def gradcam_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, last_conv_name: str = "conv5_block3_out"):
    """
    Compute Grad-CAM for ResNet50-style models.
    - img_tensor: (1, H, W, 3) preprocessed to match training
    """
    try:
        last_conv_layer = model.get_layer(last_conv_name)
    except ValueError:
        return None  # layer not found

    # Build a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of the target class wrt last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    # Global average pool the gradients over width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weight the channels by importance
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # ReLU and normalize to [0,1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to input image size
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()

    return heatmap  # float32 [0,1] (H,W)

def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """
    Overlay a grayscale heatmap onto the original RGB image.
    """
    if heatmap is None:
        return None
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_uint8, mode="L").resize(pil_img.size)
    heatmap_img = heatmap_img.convert("RGBA")
    # Convert grayscale to pseudo-color using PIL (apply a simple palette)
    heatmap_img = heatmap_img.convert("P")  # palettized
    heatmap_img.putpalette([
        # Simple palette: from black to red to yellow to white (256 entries)
        # We'll just set a gradient; PIL will interpolate
        *(list(np.linspace(0, 255, 256).astype(np.uint8)) + [0]*256 + [0]*256)
    ])
    heatmap_img = heatmap_img.convert("RGBA")

    base = pil_img.convert("RGBA")
    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended.convert("RGB")

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigator")
choice = st.sidebar.selectbox('Go to', ['Introduction', 'TB X-Ray Prediction', 'About Me'])

# ---------------------------
# Pages
# ---------------------------
if choice == 'Introduction':
    st.title('Tuberculosis X-rays Classification')
    if os.path.exists('images.jpeg'):
        st.image('images.jpeg', use_column_width=True)
    else:
        st.warning("Cover image `images.jpeg` not found in the working directory.")
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

    # Load the model (with spinner)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Place it next to `model.py`.")
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

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess
        batch = preprocess_image_pil(image, target_size=IMG_SIZE)

        # Predict
        start = time.time()
        probs = predict_with_softmax(model, batch)
        infer_time = (time.time() - start) * 1000.0  # ms

        # Decode
        class_index = int(np.argmax(probs[0]))
        class_name = CATEGORIES[class_index]
        confidence = float(np.max(probs[0])) * 100.0

        # Display
        st.success(f"**Prediction:** {class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.caption(f"Inference time: {infer_time:.1f} ms")

        # Probability table
        st.write("Class probabilities:")
        prob_table = {CATEGORIES[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(CATEGORIES))}
        st.table(prob_table)

        # Optional Grad-CAM
        if show_heatmap:
            with st.spinner("Generating Grad-CAM..."):
                heatmap = gradcam_heatmap(model, batch, last_conv_name="conv5_block3_out")
                if heatmap is None:
                    st.warning("Grad-CAM unavailable (layer not found or gradients not traceable).")
                else:
                    overlay = overlay_heatmap_on_image(image.resize((IMG_SIZE, IMG_SIZE)), heatmap, alpha=0.35)
                    st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

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

