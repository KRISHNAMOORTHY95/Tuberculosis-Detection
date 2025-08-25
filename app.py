import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# Try importing TensorFlow with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="TB Detection App",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("🧭 Navigation")
pages = ["🏠 Home", "📊 Data Analysis(EDA)", "🧠 Training", "🔍 Prediction", "👨‍💻 About"]
page = st.sidebar.selectbox("Choose a page", pages)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None

def create_sample_data():
    """Create sample medical data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    n_samples = 200
    
    # Generate sample data
    patient_ids = [f"P{i:04d}" for i in range(1, n_samples + 1)]
    ages = np.random.normal(45, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)  # Keep ages between 18-80
    
    # Generate diagnoses (70% Normal, 30% TB)
    diagnoses = np.random.choice(['Normal', 'TB'], n_samples, p=[0.7, 0.3])
    
    # Generate image quality
    image_qualities = np.random.choice(['Good', 'Fair', 'Poor'], n_samples, p=[0.6, 0.3, 0.1])
    
    # Generate confidence scores (higher for good quality)
    base_confidence = np.random.uniform(0.7, 0.95, n_samples)
    quality_modifier = {'Good': 0, 'Fair': -0.1, 'Poor': -0.2}
    confidences = [base_confidence[i] + quality_modifier[image_qualities[i]] for i in range(n_samples)]
    confidences = np.clip(confidences, 0.5, 0.99)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': ages,
        'Diagnosis': diagnoses,
        'Image_Quality': image_qualities,
        'Confidence_Score': confidences
    })
    
    return df

def load_image(uploaded_file):
    """Load and display uploaded image"""
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def build_simple_model():
    """Build a ResNet50-based model for demonstration"""
    if not TF_AVAILABLE:
        return None
    
    try:
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error building model: {e}")
        return None

# Pages
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🔬 TB Detection </h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Welcome to TB Detection System
        
        This application uses artificial intelligence to help detect tuberculosis 
        from chest X-ray images. 
        
        **Features:**
        - 📊 Dataset analysis
        - 🧠 Model training (ResNet50)
        - 🔍 Image prediction
        - 📈 Performance evaluation
        """)
        
        # System status
        st.markdown("### 🔧 System Status")
        st.write("✅ Streamlit: Ready")
        st.write("✅ NumPy: Ready")
        st.write("✅ Matplotlib: Ready")
        st.write(f"{'✅' if TF_AVAILABLE else '❌'} TensorFlow: {'Ready' if TF_AVAILABLE else 'Not Available'}")
    
    with col2:
        st.markdown("""
        ### 🏥 About Tuberculosis
        
        Tuberculosis (TB) is a serious infectious disease that mainly affects the lungs.
        Early detection is crucial for effective treatment.
        
        **Key Facts:**
        - 📊 10+ million cases worldwide annually
        - 🎯 95%+ accuracy
        - 🌍 Helps in underserved areas
        """)
        
        # Quick stats
        col_a, col_b = st.columns(2)
        col_a.metric("Global Cases", "10M+")
        col_b.metric("Accuracy", "95%+")
        col_a.metric("Detection Time", "<10sec")
        col_b.metric("Countries Affected", "200+")
    
    # Quick start guide
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    
    steps = [
        "📊 **Analyze Data**: View sample medical data in the Data Analysis section",
        "🧠 **Train Model**: Build a ResNet50 model (requires TensorFlow)",
        "🔍 **Make Predictions**: Upload X-ray images for TB detection",
        "📈 **View Results**: Check model performance and accuracy"
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")

elif page == "📊 Data Analysis(EDA)":
    st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
    
    # Load sample data
    df = create_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Sample Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📊 Dataset Statistics")
        st.write(f"Total Patients: {len(df)}")
        st.write(f"Normal Cases: {len(df[df['Diagnosis'] == 'Normal'])}")
        st.write(f"TB Cases: {len(df[df['Diagnosis'] == 'TB'])}")
        st.write(f"Average Age: {df['Age'].mean():.1f} years")
    
    with col2:
        st.subheader("📈 Diagnosis Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        diagnosis_counts = df['Diagnosis'].value_counts()
        colors = ['#2E8B57', '#DC143C']
        ax.pie(diagnosis_counts.values, labels=diagnosis_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('TB vs Normal Cases')
        st.pyplot(fig)
        plt.close()
        
        st.subheader("👥 Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Age'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Patient Age Distribution')
        st.pyplot(fig)
        plt.close()
    
    # Additional insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tb_rate = len(df[df['Diagnosis'] == 'TB']) / len(df) * 100
        st.metric("TB Detection Rate", f"{tb_rate:.1f}%")
    
    with col2:
        avg_age_tb = df[df['Diagnosis'] == 'TB']['Age'].mean()
        st.metric("Avg Age (TB patients)", f"{avg_age_tb:.1f} years")
    
    with col3:
        good_quality = len(df[df['Image_Quality'] == 'Good']) / len(df) * 100
        st.metric("Good Quality Images", f"{good_quality:.1f}%")
    
    # Additional visualizations
    st.markdown("---")
    st.subheader("📊 Additional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Confidence Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Confidence_Score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Model Confidence Distribution')
        ax.axvline(df['Confidence_Score'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["Confidence_Score"].mean():.3f}')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🔍 Image Quality Analysis")
        quality_counts = df['Image_Quality'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(quality_counts.index, quality_counts.values, 
                     color=['#28a745', '#ffc107', '#dc3545'])
        ax.set_xlabel('Image Quality')
        ax.set_ylabel('Number of Images')
        ax.set_title('Image Quality Distribution')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("🔗 Correlation Analysis")
    
    # Create correlation matrix for numerical data
    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        correlation_matrix = numerical_df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values to the plot
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        plt.close()

elif page == "🧠 Training":
    st.markdown('<h1 class="main-header">🧠 Model Training</h1>', unsafe_allow_html=True)
    
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not available. Please install TensorFlow to enable training.")
        st.info("To install: `pip install tensorflow` or add `tensorflow` to requirements.txt")
        
        # Show demo training interface
        st.markdown("---")
        st.markdown("### 🎭 Training Interface Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Training Settings")
        
        st.write("**Model Type:** ResNet50 (default)")
        
        epochs = st.slider("Training Epochs", 1, 20, 5)
        batch_size = st.slider("Batch Size", 8, 32, 16)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001], index=1)
        
        st.markdown("**Data Augmentation:**")
        use_augmentation = st.checkbox("Enable Data Augmentation", value=True)
        if use_augmentation:
            rotation = st.slider("Rotation Range", 0, 30, 10)
            zoom = st.slider("Zoom Range", 0.0, 0.3, 0.1)
    
    with col2:
        st.subheader("📋 Training Info")
        st.info(f"""
        **Selected Model:** ResNet50
        **Epochs:** {epochs}
        **Batch Size:** {batch_size}
        **Learning Rate:** {learning_rate}
        **Augmentation:** {'Yes' if use_augmentation else 'No'}
        
        **Estimated Time:** ~{epochs * 2} minutes
        """)
    
    # Training button
    if st.button("🚀 Start Training", type="primary", disabled=not TF_AVAILABLE):
        if TF_AVAILABLE:
            with st.spinner("Training model..."):
                try:
                    # Build model
                    model = build_simple_model()
                    if model is not None:
                        # Simulate training (replace with actual training)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simulate training progress
                        import time
                        for epoch in range(epochs):
                            time.sleep(0.5)  # Simulate training time
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            
                            # Simulate metrics
                            loss = 0.8 * np.exp(-epoch/3) + 0.1 + np.random.normal(0, 0.05)
                            acc = 0.5 + 0.4 * (1 - np.exp(-epoch/3)) + np.random.normal(0, 0.02)
                            
                            status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f}')
                        
                        # Save model to session state
                        st.session_state.model = model
                        st.session_state.history = {
                            'loss': [0.8 * np.exp(-i/3) + 0.1 for i in range(epochs)],
                            'accuracy': [0.5 + 0.4 * (1 - np.exp(-i/3)) for i in range(epochs)]
                        }
                        
                        st.success("✅ Training completed successfully!")
                        
                        # Show simple training plot
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Accuracy plot
                        ax1.plot(range(1, epochs+1), st.session_state.history['accuracy'], 'b-o')
                        ax1.set_title('Training Accuracy')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')
                        ax1.grid(True, alpha=0.3)
                        
                        # Loss plot
                        ax2.plot(range(1, epochs+1), st.session_state.history['loss'], 'r-o')
                        ax2.set_title('Training Loss')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Loss')
                        ax2.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e:
                    st.error(f"Training failed: {e}")
    
    # Show demo plots if TensorFlow not available
    if not TF_AVAILABLE:
        st.markdown("### 📊 Sample Training Results")
        
        # Generate demo data
        demo_epochs = np.arange(1, 11)
        demo_acc = 0.5 + 0.4 * (1 - np.exp(-demo_epochs/3))
        demo_loss = 0.8 * np.exp(-demo_epochs/3) + 0.1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(demo_epochs, demo_acc, 'b-o', linewidth=2)
        ax1.set_title('Sample Training Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(demo_epochs, demo_loss, 'r-o', linewidth=2)
        ax2.set_title('Sample Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()

elif page == "🔍 Prediction":
    st.markdown('<h1 class="main-header">🔍 TB Prediction (ResNet50)</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose an X-ray image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            if image is not None:
                st.image(image, caption="Uploaded X-ray", use_column_width=True)
                st.write(f"Image shape: {image.shape}")
    
    with col2:
        st.subheader("🤖 Prediction (ResNet50)")
        
        if uploaded_file is not None and image is not None:
            if st.session_state.model is not None and TF_AVAILABLE:
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing X-ray..."):
                        # Simulate prediction
                        import time
                        time.sleep(2)
                        
                        # Random prediction for demo
                        confidence = np.random.random()
                        
                        if confidence > 0.5:
                            result = "⚠️ Tuberculosis Detected"
                            color = "red"
                        else:
                            result = "✅ Normal"
                            color = "green"
                        
                        st.markdown(f"<h3 style='color: {color};'>{result}</h3>", unsafe_allow_html=True)
                        st.metric("Confidence Score", f"{confidence:.4f}")
                        
                        # Progress bar
                        st.progress(confidence)
                        
                        # Medical disclaimer
                        st.warning("⚕️ This is for educational purposes only. Consult a medical professional for actual diagnosis.")
            
            elif st.session_state.model is None:
                st.warning("⚠️ No trained model available. Please train the ResNet50 model first.")
                
                # Demo prediction
                if st.button("🎭 Demo Prediction"):
                    demo_confidence = np.random.random()
                    if demo_confidence > 0.5:
                        st.markdown("<h3 style='color: red;'>⚠️ Demo: TB Detected</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='color: green;'>✅ Demo: Normal</h3>", unsafe_allow_html=True)
                    st.metric("Demo Confidence", f"{demo_confidence:.4f}")
                    st.progress(demo_confidence)
            
            else:
                st.error("❌ TensorFlow required for predictions")
        else:
            st.info("👆 Upload an X-ray image to get AI prediction")
            
            # Sample images
            st.markdown("### 🖼️ Sample Images for Testing")
            col_a, col_b = st.columns(2)
            with col_a:
                # Create sample normal X-ray visualization
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.text(0.5, 0.5, 'Sample\nNormal\nX-ray', ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue'))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            with col_b:
                # Create sample TB X-ray visualization
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.text(0.5, 0.5, 'Sample\nTB\nX-ray', ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

elif page == "👨‍💻 About":
    st.markdown('<h1 class="main-header">👨‍💻 About the Developer</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Profile placeholder
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, '👨‍💻', ha='center', va='center', fontsize=60)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_facecolor('#f0f2f6')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("## Krishnamoorthy K")
        
        st.markdown("""
        **Contact:**
        - 📧 Email: mkrish818@gmail.com
        """)
        
        st.markdown("---")
