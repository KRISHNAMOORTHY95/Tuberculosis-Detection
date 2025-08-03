# -------------------------------
# STREAMLIT PAGE 1: Introduction
# -------------------------------
def show_home():
    st.title("🩺 Tuberculosis Detection Using Deep Learning")
    st.image("/home/ubuntu/TB_Project/Dataset/TB_Project_Image.png", use_container_width=True)
    st.markdown("""
        This project uses a Convolutional Neural Network with **Transfer Learning (ResNet50)** to classify chest X-rays as:
        - **Normal**
        - **Tuberculosis (TB)**

        ### 👇 Features:
        - Deep Learning with TensorFlow
        - Streamlit Web Interface
        - Image Upload & Prediction
        - AWS Deployment Ready

        ---
        """)
# -------------------------------
# STREAMLIT PAGE 2: TB Detection
# -------------------------------
def show_detection():
    st.title("🧪 TB Detection from Chest X-ray")
    model_path = '/home/ubuntu/TB_Project/models/best_model.h5'
    model = tf.keras.models.load_model(model_path)

    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        st.write(f"🧠 Model Confidence: `{pred:.2f}`")

        if pred > 0.5:
            st.error("❌ Prediction: Tuberculosis Detected")
        else:
            st.success("✅ Prediction: Normal")
# -------------------------------
# MAIN STREAMLIT APP NAVIGATION
# -------------------------------
def run_app():
    st.set_page_config(page_title="TB Detection App", layout="centered")

    menu = st.sidebar.radio("📌 Navigate", ["🏠 Home", "🧪 TB Detection", "👤 About Me"])

    if menu == "🏠 Home":
        show_home()
    elif menu == "🧪 TB Detection":
        show_detection()
    elif menu == "👤 About Me":
        show_about()

# MAIN CONTROLLER
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TB Detection Project")
    parser.add_argument('--prepare', action='store_true', help="Step 1: Prepare dataset")
    parser.add_argument('--train', action='store_true', help="Step 2: Train model")
    parser.add_argument('--evaluate', action='store_true', help="Step 3: Evaluate model")
    parser.add_argument('--app', action='store_true', help="Step 4: Launch Streamlit app")

    # In a Colab environment, sys.argv will contain extra arguments related to the kernel.
    # We only want to parse arguments that are explicitly provided by the user for our script.
    # We check if we are in a known interactive environment like ipykernel (used by Colab)
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([]) # Pass an empty list to parse_args
    else:
        args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
    elif args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.app:
        run_app()
    else:
        print("📌 Usage:\n"
              "  python main.py --prepare   # Prepare dataset\n"
              "  python main.py --train     # Train model\n"
              "  python main.py --evaluate  # Evaluate model\n"
              "  streamlit run main.py -- --app  # Run Streamlit app")
