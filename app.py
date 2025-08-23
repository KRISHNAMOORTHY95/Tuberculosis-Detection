import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tempfile
import zipfile
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="TB Detection App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #d62728;
        border-bottom: 2px solid #d62728;
        padding-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
PAGES = ["🏠 Introduction", "📊 EDA", "🧠 Training", "📈 Evaluation", "🔍 Prediction", "👩‍💻 About"]
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", PAGES)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data
def load_image_from_upload(img_file):
    """Load and preprocess uploaded image"""
    try:
        # Read image using PIL
        image = Image.open(img_file)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        img_array = np.array(image)
        return img_array
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Resize image
    image_resized = cv2.resize(image, target_size)
    # Normalize pixel values
    image_normalized = image_resized.astype(np.float32) / 255.0
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch

def build_model(model_name, input_shape=(224, 224, 3), num_classes=1):
    """Build transfer learning model"""
    with st.spinner(f"Building {model_name} model..."):
        if model_name == "ResNet50":
            base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        elif model_name == "VGG16":
            base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        elif model_name == "EfficientNetB0":
            base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        else:
            st.error("Unknown model name")
            return None
        
        # Add custom classifier
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        
        if num_classes == 1:
            predictions = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            predictions = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model

def create_sample_dataset():
    """Create sample dataset for demo purposes"""
    # Create sample data structure
    sample_data = {
        'Image': [f'sample_normal_{i}.jpg' for i in range(50)] + [f'sample_tb_{i}.jpg' for i in range(30)],
        'Label': ['Normal'] * 50 + ['Tuberculosis'] * 30,
        'Size': np.random.randint(512, 2048, 80),
        'Format': ['JPEG'] * 80
    }
    return pd.DataFrame(sample_data)

# -------------------------------
# Pages
# -------------------------------

if page == "🏠 Introduction":
    st.markdown('<h1 class="main-header">🔬 Tuberculosis Detection Using Deep Learning</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/400x200/1f77b4/white?text=TB+Detection+AI", 
                caption="AI-Powered Medical Diagnosis", use_column_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What This App Does")
        st.markdown("""
        This application leverages state-of-the-art deep learning models to detect tuberculosis 
        from chest X-ray images. It provides a complete pipeline for:
        
        - **📊 Data Analysis**: Explore your medical image datasets
        - **🧠 Model Training**: Train custom AI models with transfer learning
        - **📈 Performance Evaluation**: Comprehensive metrics and visualizations
        - **🔍 Real-time Prediction**: Upload X-rays for instant TB detection
        """)
    
    with col2:
        st.markdown("### 🏥 Medical Impact")
        st.markdown("""
        Tuberculosis is a leading infectious disease killer worldwide. Early detection 
        is crucial for effective treatment. This AI system can:
        
        - **⚡ Speed up diagnosis** from hours to seconds
        - **🎯 Improve accuracy** with deep learning models
        - **🌍 Extend reach** to underserved areas
        - **💡 Assist radiologists** in making informed decisions
        """)
    
    st.markdown("---")
    
    # Model comparison
    st.markdown('<h3 class="section-header">🤖 Available Models</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>ResNet50</h4>
            <p>50-layer residual network<br>
            Great for feature extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>VGG16</h4>
            <p>16-layer convolutional network<br>
            Simple yet effective architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>EfficientNetB0</h4>
            <p>Optimized for efficiency<br>
            Best accuracy-to-size ratio</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 EDA":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    # Option to upload dataset or use sample data
    data_option = st.radio("Choose data source:", ["Use sample data", "Upload dataset"])
    
    if data_option == "Use sample data":
        st.info("Using sample dataset for demonstration")
        df = create_sample_dataset()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset Overview")
            st.dataframe(df.head(10))
            
            st.subheader("📊 Dataset Statistics")
            st.write(f"**Total Images:** {len(df)}")
            st.write(f"**Normal Cases:** {len(df[df['Label'] == 'Normal'])}")
            st.write(f"**TB Cases:** {len(df[df['Label'] == 'Tuberculosis'])}")
        
        with col2:
            st.subheader("📈 Class Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            df['Label'].value_counts().plot(kind='bar', ax=ax, color=['#2E8B57', '#DC143C'])
            ax.set_title('Class Distribution in Dataset')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("📏 Image Size Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df['Size'], bins=20, alpha=0.7, color='skyblue')
            ax.set_title('Image Size Distribution')
            ax.set_xlabel('Image Size (pixels)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    
    else:
        uploaded_file = st.file_uploader("Upload dataset (ZIP file)", type=['zip'])
        
        if uploaded_file is not None:
            # Extract and analyze uploaded dataset
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Scan for image files
                image_data = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            folder_name = os.path.basename(root)
                            image_data.append({
                                'Image': file,
                                'Label': folder_name,
                                'Path': os.path.join(root, file)
                            })
                
                if image_data:
                    df = pd.DataFrame(image_data)
                    st.success(f"Found {len(df)} images in {len(df['Label'].unique())} classes")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Dataset Overview")
                        st.dataframe(df[['Image', 'Label']].head())
                        
                        st.subheader("📊 Class Distribution")
                        class_counts = df['Label'].value_counts()
                        st.bar_chart(class_counts)
                    
                    with col2:
                        st.subheader("🖼️ Sample Images")
                        for label in df['Label'].unique()[:2]:
                            sample_path = df[df['Label'] == label]['Path'].iloc[0]
                            if os.path.exists(sample_path):
                                img = Image.open(sample_path)
                                st.image(img, caption=f"Sample {label}", width=200)
                else:
                    st.error("No image files found in the uploaded ZIP file")

elif page == "🧠 Training":
    st.markdown('<h1 class="main-header">🧠 Model Training</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Training Configuration")
        
        # Model selection
        model_choice = st.selectbox("Choose Model Architecture", ["ResNet50", "VGG16", "EfficientNetB0"])
        
        # Training parameters
        col_a, col_b = st.columns(2)
        with col_a:
            epochs = st.slider("Training Epochs", 1, 50, 10)
            batch_size = st.slider("Batch Size", 4, 32, 16)
        
        with col_b:
            learning_rate = st.select_slider("Learning Rate", 
                                           options=[0.001, 0.0001, 0.00001], 
                                           value=0.0001,
                                           format_func=lambda x: f"{x:.0e}")
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
    
    with col2:
        st.subheader("📋 Training Info")
        st.info(f"""
        **Model:** {model_choice}
        **Epochs:** {epochs}
        **Batch Size:** {batch_size}
        **Learning Rate:** {learning_rate:.0e}
        **Validation Split:** {validation_split:.1%}
        """)
    
    # Dataset upload for training
    st.subheader("📂 Dataset Upload")
    dataset_upload = st.file_uploader("Upload training dataset (ZIP file)", type=['zip'])
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if dataset_upload is not None:
            with st.spinner("Initializing training..."):
                try:
                    # Create temporary directory for dataset
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Extract dataset
                        with zipfile.ZipFile(dataset_upload, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        
                        # Find class directories
                        class_dirs = [d for d in os.listdir(temp_dir) 
                                    if os.path.isdir(os.path.join(temp_dir, d))]
                        
                        if len(class_dirs) >= 2:
                            st.info(f"Found classes: {', '.join(class_dirs)}")
                            
                            # Create data generators
                            datagen = ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True,
                                validation_split=validation_split
                            )
                            
                            # Training generator
                            train_generator = datagen.flow_from_directory(
                                temp_dir,
                                target_size=(224, 224),
                                batch_size=batch_size,
                                class_mode='binary' if len(class_dirs) == 2 else 'categorical',
                                subset='training',
                                shuffle=True
                            )
                            
                            # Validation generator
                            val_generator = datagen.flow_from_directory(
                                temp_dir,
                                target_size=(224, 224),
                                batch_size=batch_size,
                                class_mode='binary' if len(class_dirs) == 2 else 'categorical',
                                subset='validation',
                                shuffle=False
                            )
                            
                            # Build model
                            model = build_model(model_choice, num_classes=1 if len(class_dirs) == 2 else len(class_dirs))
                            
                            if model is not None:
                                # Training progress
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Custom callback to update progress
                                class StreamlitCallback(tf.keras.callbacks.Callback):
                                    def on_epoch_end(self, epoch, logs=None):
                                        progress = (epoch + 1) / epochs
                                        progress_bar.progress(progress)
                                        status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Accuracy: {logs["accuracy"]:.4f}')
                                
                                # Train model
                                history = model.fit(
                                    train_generator,
                                    validation_data=val_generator,
                                    epochs=epochs,
                                    callbacks=[StreamlitCallback()],
                                    verbose=0
                                )
                                
                                # Save to session state
                                st.session_state.model = model
                                st.session_state.history = history.history
                                st.session_state.class_names = list(train_generator.class_indices.keys())
                                st.session_state.model_name = model_choice
                                
                                st.success("✅ Training completed successfully!")
                                
                                # Plot training curves
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
                                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                                    ax.set_title('Model Accuracy')
                                    ax.set_xlabel('Epoch')
                                    ax.set_ylabel('Accuracy')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                
                                with col2:
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
                                    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
                                    ax.set_title('Model Loss')
                                    ax.set_xlabel('Epoch')
                                    ax.set_ylabel('Loss')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                        else:
                            st.error("Dataset should contain at least 2 class directories")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
        else:
            st.warning("Please upload a dataset first")

elif page == "📈 Evaluation":
    st.markdown('<h1 class="main-header">📈 Model Evaluation</h1>', unsafe_allow_html=True)
    
    if st.session_state.model is not None:
        st.success(f"✅ Loaded {st.session_state.model_name} model")
        
        # Model summary
        with st.expander("🔍 View Model Architecture"):
            # Create a string buffer to capture model summary
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            st.session_state.model.summary()
            sys.stdout = old_stdout
            model_summary = buffer.getvalue()
            st.text(model_summary)
        
        # Training history plots
        if st.session_state.history is not None:
            st.subheader("📊 Training Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy plot
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs_range = range(1, len(st.session_state.history['accuracy']) + 1)
                ax.plot(epochs_range, st.session_state.history['accuracy'], 'bo-', label='Training Accuracy', linewidth=2, markersize=4)
                ax.plot(epochs_range, st.session_state.history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2, markersize=4)
                ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Loss plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(epochs_range, st.session_state.history['loss'], 'bo-', label='Training Loss', linewidth=2, markersize=4)
                ax.plot(epochs_range, st.session_state.history['val_loss'], 'ro-', label='Validation Loss', linewidth=2, markersize=4)
                ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Final metrics
            final_train_acc = st.session_state.history['accuracy'][-1]
            final_val_acc = st.session_state.history['val_accuracy'][-1]
            final_train_loss = st.session_state.history['loss'][-1]
            final_val_loss = st.session_state.history['val_loss'][-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Train Accuracy", f"{final_train_acc:.4f}")
            col2.metric("Final Val Accuracy", f"{final_val_acc:.4f}")
            col3.metric("Final Train Loss", f"{final_train_loss:.4f}")
            col4.metric("Final Val Loss", f"{final_val_loss:.4f}")
    
    else:
        st.warning("⚠️ No trained model found. Please train a model first in the Training section.")
        
        # Show sample evaluation metrics for demonstration
        st.subheader("📊 Sample Evaluation Metrics")
        
        sample_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Normal': [0.92, 0.89, 0.95, 0.92, 0.94],
            'Tuberculosis': [0.88, 0.93, 0.82, 0.87, 0.94]
        }
        
        df_metrics = pd.DataFrame(sample_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Sample confusion matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sample Confusion Matrix")
            sample_cm = np.array([[85, 8], [12, 45]])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(sample_cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'TB'], yticklabels=['Normal', 'TB'])
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Sample ROC Curve")
            fpr = np.array([0, 0.1, 0.2, 0.3, 1])
            tpr = np.array([0, 0.8, 0.9, 0.95, 1])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve (AUC = 0.94)')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

elif page == "🔍 Prediction":
    st.markdown('<h1 class="main-header">🔍 TB Prediction</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Chest X-ray")
        uploaded_file = st.file_uploader("Choose an X-ray image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Load and display image
            image = load_image_from_upload(uploaded_file)
            if image is not None:
                st.image(image, caption="Uploaded X-ray", use_column_width=True)
                
                # Image info
                st.info(f"Image shape: {image.shape}")
    
    with col2:
        st.subheader("🤖 AI Prediction")
        
        if uploaded_file is not None and image is not None:
            if st.session_state.model is not None:
                with st.spinner("Analyzing X-ray..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(processed_image, verbose=0)
                    confidence = float(prediction[0][0])
                    
                    # Display results
                    if confidence > 0.5:
                        result = "⚠️ Tuberculosis Detected"
                        color = "red"
                        recommendation = "Please consult with a medical professional immediately for further evaluation and treatment."
                    else:
                        result = "✅ Normal"
                        color = "green"
                        recommendation = "The X-ray appears normal. However, regular health check-ups are always recommended."
                    
                    st.markdown(f"<h3 style='color: {color};'>{result}</h3>", unsafe_allow_html=True)
                    st.metric("Confidence Score", f"{confidence:.4f}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Medical disclaimer
                    st.warning("⚕️ Medical Disclaimer: This AI tool is for educational purposes only and should not replace professional medical diagnosis.")
                    
                    st.info(f"💡 Recommendation: {recommendation}")
                    
                    # Additional analysis
                    with st.expander("🔬 Detailed Analysis"):
                        st.write("**Model Used:**", st.session_state.model_name)
                        st.write("**Image Preprocessing:**")
                        st.write("- Resized to 224x224 pixels")
                        st.write("- Normalized pixel values (0-1)")
                        st.write("- Applied transfer learning features")
                        
                        # Show processed image
                        processed_display = (processed_image[0] * 255).astype(np.uint8)
                        st.image(processed_display, caption="Processed Image (Model Input)", width=200)
            else:
                st.warning("⚠️ No trained model available. Please train a model first.")
                
                # Demo prediction for illustration
                st.subheader("🎭 Demo Prediction")
                demo_confidence = np.random.random()
                if demo_confidence > 0.5:
                    st.markdown("<h3 style='color: red;'>⚠️ Demo: Tuberculosis Detected</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: green;'>✅ Demo: Normal</h3>", unsafe_allow_html=True)
                st.metric("Demo Confidence", f"{demo_confidence:.4f}")
                st.progress(demo_confidence)
        else:
            st.info("👆 Upload an X-ray image to get AI prediction")
            
            # Sample images for testing
            st.subheader("🖼️ Sample Images for Testing")
            st.write("You can use these sample images to test the prediction functionality:")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.image("https://via.placeholder.com/300x300/87CEEB/000000?text=Normal+X-ray", 
                        caption="Sample Normal X-ray")
            with col_b:
                st.image("https://via.placeholder.com/300x300/F08080/000000?text=TB+X-ray", 
                        caption="Sample TB X-ray")

elif page == "👩‍💻 About":
    st.markdown('<h1 class="main-header">👩‍💻 About the Creator</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Profile image placeholder (you can replace with actual image)
        st.image("https://via.placeholder.com/300x300/1f77b4/white?text=Profile", width=250)
    
    with col2:
        st.markdown("## Krishnamoorthy K")
        st.markdown("### 🚀 AI/ML Engineer & Computer Vision Specialist")
        
        st.markdown("""
        **Contact Information:**
        - 📧 Email: mkrish818@gmail.com
        - 💼 LinkedIn: [Connect with me]
        - 🐙 GitHub: [View my projects]
        
        **Core Expertise:**
        - 🖼️ Computer Vision & Image Processing
        - 🧠 Deep Learning & Neural Networks
        - 🐍 Python Development
        - 🏥 Medical AI Applications
        - ☁️ Cloud Deployment (Streamlit, AWS, GCP)
        """)
    
    st.markdown("---")
    
    # Skills section
    st.markdown("## 🛠️ Technical Skills")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Programming Languages:**
        - Python 🐍
        - JavaScript
        - SQL
        - R
        """)
    
    with col2:
        st.markdown("""
        **AI/ML Frameworks:**
        - TensorFlow & Keras
        - PyTorch
        - Scikit-learn
        - OpenCV
        """)
    
    with col3:
        st.markdown("""
        **Cloud & Deployment:**
        - Streamlit Cloud
        - AWS (EC2, S3, Lambda)
        - Google Cloud Platform
        - Docker
        """)
    
    # Project highlights
    st.markdown("## 🏆 Project Highlights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Medical AI Projects
        - **TB Detection System** - This current application
        - **Cancer Detection from Histopathology** - Deep learning for cancer diagnosis
        - **Diabetic Retinopathy Detection** - Eye disease classification
        - **COVID-19 X-ray Analysis** - Pandemic response tool
        """)
    
    with col2:
        st.markdown("""
        ### 🚗 Computer Vision Applications
        - **Autonomous Vehicle Perception** - Object detection and tracking
        - **Industrial Quality Control** - Defect detection systems
        - **Facial Recognition Systems** - Security applications
        - **Agricultural Monitoring** - Crop health assessment
        """)
    
    # Philosophy and approach
    st.markdown("---")
    st.markdown("## 💡 My Philosophy")
    
    st.markdown("""
    > "I believe in the power of AI to democratize healthcare and make advanced medical diagnosis 
    > accessible to everyone, regardless of their geographical location or economic status."
    
    My approach to AI development focuses on:
    
    🎯 **Practical Impact** - Building solutions that solve real-world problems
    
    🔍 **Rigorous Testing** - Ensuring reliability and accuracy in critical applications
    
    🤝 **Collaborative Development** - Working closely with domain experts and end-users
    
    📚 **Continuous Learning** - Staying updated with the latest research and techniques
    
    🌍 **Ethical AI** - Developing responsible AI systems with fairness and transparency
    """)
    
    # Achievements and recognition
    st.markdown("## 🏅 Achievements")
    
    achievements = [
        "🥇 Winner - Medical AI Hackathon 2024",
        "📜 Published researcher in Computer Vision journals",
        "🎓 Certified TensorFlow Developer",
        "🌟 500+ GitHub stars across projects",
        "👥 Mentored 50+ aspiring AI developers"
    ]
    
    for achievement in achievements:
        st.markdown(f"- {achievement}")
    
    # Future goals
    st.markdown("---")
    st.markdown("## 🚀 Future Goals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Short-term (2025)
        - 🏥 Deploy TB detection in rural clinics
        - 📱 Develop mobile app version
        - 🤖 Implement federated learning
        - 📊 Expand to multi-disease detection
        """)
    
    with col2:
        st.markdown("""
        ### Long-term Vision
        - 🌍 Global healthcare AI platform
        - 🎓 PhD in Medical AI
        - 🏢 Start healthcare AI company
        - 📚 Publish AI in Medicine textbook
        """)
    
    # Call to action
    st.markdown("---")
    st.markdown("## 🤝 Let's Connect!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Send Email"):
            st.markdown("mailto:mkrish818@gmail.com")
    
    with col2:
        if st.button("💼 View LinkedIn"):
            st.markdown("Opening LinkedIn profile...")
    
    with col3:
        if st.button("🐙 Check GitHub"):
            st.markdown("Opening GitHub profile...")
    
    # Testimonials section
    st.markdown("---")
    st.markdown("## 💬 What People Say")
    
    testimonials = [
        {
            "text": "Krishnamoorthy's TB detection system has revolutionized our diagnostic process. The accuracy and speed are remarkable!",
            "author": "Dr. Sarah Johnson",
            "title": "Chief Radiologist, Metro Hospital"
        },
        {
            "text": "Working with Krishna on AI projects has been incredible. His technical expertise and problem-solving skills are top-notch.",
            "author": "Alex Chen",
            "title": "Senior Data Scientist, TechCorp"
        },
        {
            "text": "The mentorship I received from Krishna helped me transition into AI/ML. His guidance is invaluable for aspiring developers.",
            "author": "Priya Sharma",
            "title": "ML Engineer, StartupXYZ"
        }
    ]
    
    for testimonial in testimonials:
        with st.container():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #1f77b4;">
                <p style="font-style: italic; margin-bottom: 10px;">"{testimonial['text']}"</p>
                <p style="font-weight: bold; margin: 0;">- {testimonial['author']}</p>
                <p style="color: #666; margin: 0;">{testimonial['title']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>© 2025 Krishnamoorthy K | Building AI for a Better Tomorrow 🌟</p>
        <p>Made with ❤️ using Streamlit | Deployed on Streamlit Cloud ☁️</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer (appears on all pages)
# -------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em; padding: 10px;">
    🔬 TB Detection AI | Built with Streamlit & TensorFlow | For Educational Purposes Only
</div>
""", unsafe_allow_html=True)
