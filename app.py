import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Configure page settings
st.set_page_config(
    page_title="TB X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .normal-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .tb-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
choice = st.sidebar.selectbox('Navigator', ['Introduction', 'TB X-Ray Prediction', 'About Me'])

if choice == 'Introduction':
    st.markdown('<h1 class="main-header">🫁 Tuberculosis X-rays Classification</h1>', unsafe_allow_html=True)
    
    # Check if image exists, use placeholder if not
    try:
        st.image('images.jpeg', use_column_width=True)
    except:
        st.info("📸 Image file 'images.jpeg' not found. Please ensure the image is in the correct directory.")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This AI-powered system leverages deep learning to analyze chest X-ray images for tuberculosis detection. 
    The application provides:
    
    - **Automated TB Detection**: Uses a trained ResNet50 model for accurate classification
    - **User-Friendly Interface**: Simple upload and instant results
    - **Medical Support Tool**: Assists healthcare professionals in preliminary screening
    
    ### 🔬 How It Works
    
    1. **Image Preprocessing**: Uploaded X-rays are resized and normalized
    2. **Deep Learning Analysis**: ResNet50 model analyzes the image features
    3. **Classification**: Returns probability scores for Normal vs Tuberculosis
    4. **Results Display**: Shows prediction with confidence percentage
    
    ### ⚠️ Important Disclaimer
    
    This tool is designed for educational and research purposes. It should **NOT** replace professional medical diagnosis. 
    Always consult qualified healthcare professionals for medical decisions.
    """)

elif choice == 'TB X-Ray Prediction':
    MODEL_PATH = 'tb_classifier_resnet50.keras'
    IMG_SIZE = 224
    CATEGORIES = ['Normal', 'Tuberculosis']

    # Load Model with better error handling
    @st.cache_resource
    def load_model(path):
        """Loads the pre-trained Keras model with comprehensive error handling."""
        if not os.path.exists(path):
            st.error(f"❌ Model file '{path}' not found. Please ensure the model is in the correct directory.")
            return None
        
        try:
            model = tf.keras.models.load_model(path)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.info("Please check if the model file is valid and compatible with the current TensorFlow version.")
            return None

    def preprocess_image(image, img_size):
        """Preprocess the uploaded image for model prediction."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image_resized = image.resize((img_size, img_size))
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        return img_array

    # UI Elements
    st.title("🫁 TB X-Ray Image Classification")
    st.write("Upload a chest X-ray image to classify it as Normal or containing Tuberculosis.")
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear chest X-ray image in JPG, JPEG, or PNG format"
        )
    
    with col2:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
            
            # Show image details
            st.write(f"**Image size:** {image.size}")
            st.write(f"**Image mode:** {image.mode}")

    # Load model
    model = load_model(MODEL_PATH)
    
    if uploaded_file is not None and model is not None:
        # Prediction section
        st.markdown("---")
        st.subheader("🔍 Analysis Results")
        
        try:
            # Preprocess image
            img_array = preprocess_image(image, IMG_SIZE)
            
            # Make prediction
            with st.spinner('🧠 Analyzing X-ray image...'):
                prediction = model.predict(img_array, verbose=0)
                score = tf.nn.softmax(prediction[0])
                class_index = np.argmax(score)
                class_name = CATEGORIES[class_index]
                confidence = float(np.max(score) * 100)
            
            # Display results with styling
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                if class_name == 'Normal':
                    st.markdown(f"""
                    <div class="prediction-box normal-result">
                        <h3>✅ Prediction: {class_name}</h3>
                        <h4>Confidence: {confidence:.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box tb-result">
                        <h3>⚠️ Prediction: {class_name}</h3>
                        <h4>Confidence: {confidence:.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                # Show probability distribution
                st.subheader("📊 Probability Distribution")
                prob_data = {
                    'Class': CATEGORIES,
                    'Probability': [float(score[i] * 100) for i in range(len(CATEGORIES))]
                }
                st.bar_chart(prob_data, x='Class', y='Probability')
            
            # Interpretation and recommendations
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if confidence > 80:
                confidence_level = "High"
                confidence_color = "🟢"
            elif confidence > 60:
                confidence_level = "Medium"
                confidence_color = "🟡"
            else:
                confidence_level = "Low"
                confidence_color = "🔴"
            
            st.write(f"{confidence_color} **Confidence Level:** {confidence_level}")
            
            if class_name == 'Tuberculosis':
                st.warning("""
                ⚠️ **Important:** This result suggests possible tuberculosis indicators. 
                Please consult a healthcare professional immediately for proper medical evaluation and testing.
                """)
            else:
                st.info("""
                ℹ️ The analysis suggests normal lung appearance. However, this tool is not a substitute 
                for professional medical diagnosis. Regular health check-ups are recommended.
                """)
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Please try uploading a different image or check the image format.")

elif choice == 'About Me':
    st.title('👩‍💻 Creator Info')
    
    # Check if about image exists
    try:
        st.image('AboutMe.webp', width=300)
    except:
        st.info("📸 Profile image 'AboutMe.webp' not found.")
    
    # Professional profile with better formatting
    st.markdown("""
    ## 🚀 Developer Profile
    
    **Name:** Krishnamoorthy K  
    **Email:** mkrish818@gmail.com  
    
    ### 🛠️ Technical Skills
    - **Computer Vision** - Image processing and analysis
    - **Deep Learning** - Neural networks and model development  
    - **Python** - Data science and machine learning
    - **TensorFlow/Keras** - Deep learning frameworks
    - **Streamlit** - Web application development
    
    ### 💡 About Me
    I am passionate about leveraging artificial intelligence to solve real-world problems, 
    particularly in healthcare and medical imaging. I enjoy learning new technologies and 
    adapting quickly to evolving environments in the field of AI and machine learning.
    
    ### 🎯 Project Goals
    This TB classification system demonstrates the potential of AI in medical screening, 
    aiming to assist healthcare professionals in early detection and diagnosis.
    
    ---
    *Feel free to reach out for collaborations or discussions about AI in healthcare!*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🔬 TB X-Ray Classification System | Built with Streamlit & TensorFlow</p>
    <p>⚠️ For educational and research purposes only - Not for clinical diagnosis</p>
</div>
""", unsafe_allow_html=True)
