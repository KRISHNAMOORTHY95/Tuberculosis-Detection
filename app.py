import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os

# Page configuration
st.set_page_config(
    page_title="TB X-ray Classifier",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar navigation with better styling
st.sidebar.title("🧭 Navigation")
choice = st.sidebar.selectbox(
    'Choose a page:', 
    ['🏠 Introduction', '🔬 TB X-Ray Prediction', '👤 About Me']
)

# Add helpful info in sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** For best results, use clear chest X-ray images "
    "with good contrast and minimal artifacts."
)

if choice == '🏠 Introduction':
    st.title('🫁 Tuberculosis X-rays Classification')
    
    # Check if image exists, show placeholder if not
    if os.path.exists('images.jpeg'):
        st.image('images.jpeg', use_container_width=True)
    else:
        st.image(
            'https://upload.wikimedia.org/wikipedia/commons/8/8c/Chest_X-ray_PA_1.jpg',
            caption='Tuberculosis Detection from X-rays (Demo Image)',
            use_container_width=True
        )
    
    st.subheader(
        'AI-Powered Tuberculosis Detection System'
    )
    
    st.markdown("""
    ### 🎯 **What This System Does:**
    This system uses deep learning to analyze chest X-ray images and help detect tuberculosis. 
    The system preprocesses and augments image data, utilizes pre-trained deep learning models, 
    and provides an intuitive interface for medical image analysis.
    
    ### ✨ **Key Features:**
    - **🧠 Deep Learning:** Powered by ResNet50 architecture
    - **⚡ Fast Processing:** Real-time image analysis
    - **📊 Confidence Scores:** Detailed probability analysis
    - **🎯 High Accuracy:** Trained on medical imaging data
    - **🔒 Secure:** No data stored or transmitted
    
    ### 📋 **How to Use:**
    1. Navigate to the **TB X-Ray Prediction** page
    2. Upload a chest X-ray image (JPG, JPEG, or PNG)
    3. Wait for the analysis
    4. Review the prediction and confidence score
    
    ### ⚠️ **Important Disclaimer:**
    This tool is for educational and research purposes only. 
    Always consult healthcare professionals for medical diagnosis.
    """)
    
    # Add some metrics for visual appeal
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "ResNet50")
    with col2:
        st.metric("Classes", "2")
    with col3:
        st.metric("Input Size", "224x224")

elif choice == '🔬 TB X-Ray Prediction':
    MODEL_PATH = '/content/tb_classifier_resnet50.keras'
    IMG_SIZE = 224
    CATEGORIES = ['Normal', 'Tuberculosis']
    
    # Enhanced model loading with better error handling
    @st.cache_resource(show_spinner=True)
    def load_model(path):
        """Loads the pre-trained Keras model with enhanced error handling."""
        try:
            uploaded_model = st.file_uploader("Upload tb_classifier_resnet50.keras", type=["keras", "h5"])
            if uploaded_model:
                with open("tb_classifier_resnet50.keras", "wb") as f:
                    f.write(uploaded_model.getbuffer())
            MODEL_PATH = "tb_classifier_resnet50.keras"
            
            model = tf.keras.models.load_model(path)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

    def validate_image(image):
        """Basic image validation."""
        width, height = image.size
        if width < 50 or height < 50:
            st.warning("⚠️ Image seems very small. Results may not be accurate.")
            return False
        return True

    def preprocess_image(image, target_size):
        """Enhanced image preprocessing."""
        # Resize with high-quality resampling
        image_resized = image.resize((target_size, target_size), Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
        img_array = img_array.astype('float32') / 255.0  # Normalize to [0,1]
        
        return img_array

    # UI Elements
    st.title("🔬 TB X-Ray Image Classification")
    st.markdown("""
    Upload a chest X-ray image to get an AI-powered analysis for tuberculosis detection.
    The system will analyze the image and provide a prediction with confidence scores.
    """)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    if model is not None:
        # Display model info
        with st.expander("🔍 Model Information"):
            st.write(f"**Model Architecture:** {type(model).__name__}")
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
            st.write(f"**Total Parameters:** {model.count_params():,}")
        
        # File uploader with better styling
        uploaded_file = st.file_uploader(
            "📁 Choose a chest X-ray image...", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Maximum file size: 200MB"
        )

        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file).convert('RGB')
                
                # Validate image
                if not validate_image(image):
                    st.stop()
                
                st.image(image, caption='📸 Uploaded X-ray Image', use_container_width=True)
                
                # Show image details
                with st.expander("📊 Image Details"):
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**Size:** {image.size}")
                    st.write(f"**Mode:** {image.mode}")
                    st.write(f"**File Size:** {uploaded_file.size:,} bytes")

                # Preprocess image
                with st.spinner('🔄 Preprocessing image...'):
                    img_array = preprocess_image(image, IMG_SIZE)

                # Make prediction with timing
                with st.spinner('🧠 AI is analyzing the X-ray...'):
                    start_time = time.time()
                    prediction = model.predict(img_array, verbose=0)
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Calculate probabilities and confidence
                    score = tf.nn.softmax(prediction[0])
                    class_index = np.argmax(score)
                    class_name = CATEGORIES[class_index]
                    confidence = 100 * np.max(score)
                    uncertainty = 1 - np.max(score)

                # Display results with better formatting
                st.markdown("## 📊 Analysis Results")
                
                # Main prediction with color coding
                if class_name == "Tuberculosis":
                    st.error(f"🚨 **Prediction:** {class_name}")
                else:
                    st.success(f"✅ **Prediction:** {class_name}")

                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.metric("Uncertainty", f"{uncertainty:.3f}")
                with col3:
                    st.metric("Inference Time", f"{inference_time:.0f}ms")
                with col4:
                    confidence_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
                    st.metric("Confidence Level", confidence_level)

                # Detailed probability breakdown
                st.markdown("### 🎯 Detailed Probabilities")
                
                for i, category in enumerate(CATEGORIES):
                    prob = float(score[i]) * 100
                    is_predicted = i == class_index
                    
                    # Create progress bar with custom styling
                    if is_predicted:
                        st.markdown(f"**🎯 {category}:** {prob:.2f}%")
                    else:
                        st.markdown(f"**{category}:** {prob:.2f}%")
                    
                    st.progress(prob / 100)
                    st.write("")  # Add spacing

                # Clinical interpretation
                st.markdown("### 🏥 Clinical Interpretation")
                
                if confidence > 80:
                    if class_name == "Tuberculosis":
                        st.warning(
                            "⚠️ **High confidence TB detection.** "
                            "This result suggests possible tuberculosis. "
                            "Immediate medical consultation is strongly recommended."
                        )
                    else:
                        st.info(
                            "ℹ️ **High confidence normal classification.** "
                            "The X-ray appears normal with high confidence. "
                            "Regular health check-ups are still recommended."
                        )
                else:
                    st.info(
                        "ℹ️ **Moderate/Low confidence prediction.** "
                        "The AI is less certain about this result. "
                        "Additional medical imaging or consultation is recommended."
                    )

                # Disclaimer
                st.markdown("---")
                st.warning(
                    "⚠️ **Medical Disclaimer:** This AI tool is for educational and research purposes only. "
                    "It should not be used as a substitute for professional medical diagnosis. "
                    "Always consult qualified healthcare professionals for medical advice."
                )

            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("Please try uploading a different image or check the file format.")

elif choice == '👤 About Me':
    st.title('👩‍💻 About the Creator')
    
    # Profile image with fallback
    if os.path.exists('AboutMe.webp'):
        st.image('AboutMe.webp', width=200)
    else:
        st.info("💼 Profile image not available")
    
    st.markdown("""
    ### Krishnamoorthy
        
    📧 **Contact:** mkrish818@gmail.com
    
    ### 🛠️ **Technical Expertise**
    - **Computer Vision:** Medical image analysis, Deep learning architectures
    - **Deep Learning:** TensorFlow, Keras, PyTorch
    - **Python Development:** Streamlit, Flask, Data Science libraries
    - **Healthcare AI:** Medical imaging, Diagnostic systems
    
    ### 🌟 **Professional Qualities**
    - **Quick Learner:** Rapidly adapts to new technologies and methodologies
    - **Problem Solver:** Passionate about solving real-world challenges with AI
    - **Innovation-Driven:** Constantly exploring cutting-edge solutions
    - **Detail-Oriented:** Ensures high-quality, reliable implementations
    
    ### 🎯 **Mission**
    Developing AI-powered solutions that make healthcare more accessible and accurate. 
    I believe technology can bridge gaps in medical diagnosis, especially in regions 
    where specialized healthcare resources are limited.
    
    ### 🚀 **Current Focus**
    - Medical image analysis for early disease detection
    - Democratizing AI tools for healthcare applications
    - Building user-friendly interfaces for complex AI systems
    
    ---
    
    *"Passionate about leveraging AI to create meaningful impact in healthcare and beyond!"*
    
    ### 🤝 **Let's Connect**
    Feel free to reach out for collaborations, any questions about this project!
    """)
    




