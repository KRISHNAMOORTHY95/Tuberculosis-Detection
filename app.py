import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Tuberculosis Detection",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------
# Session State Initialization
# ---------------------------
if "model" not in st.session_state:
    st.session_state["model"] = None
if "history" not in st.session_state:
    st.session_state["history"] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = False

# ---------------------------
# Helper: Create Sample Data
# ---------------------------
def create_sample_data():
    """Generate synthetic sample patient data for demo"""
    np.random.seed(42)
    n = 200
    data = {
        "Patient_ID": [f"P{i:04d}" for i in range(1, n+1)],
        "Age": np.random.randint(10, 80, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Diagnosis": np.random.choice(["Normal", "TB"], n, p=[0.6, 0.4]),
        "Image_Quality": np.random.choice(["Good", "Poor"], n, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    return df

# ---------------------------
# Helper: Simulated Training
# ---------------------------
def train_model():
    """Simulate model training and save history"""
    epochs = 10
    acc = np.linspace(0.6, 0.95, epochs) + np.random.normal(0, 0.02, epochs)
    val_acc = np.linspace(0.55, 0.92, epochs) + np.random.normal(0, 0.02, epochs)
    loss = np.linspace(1.2, 0.2, epochs) + np.random.normal(0, 0.05, epochs)
    val_loss = np.linspace(1.3, 0.3, epochs) + np.random.normal(0, 0.05, epochs)

    history = {
        "accuracy": acc,
        "val_accuracy": val_acc,
        "loss": loss,
        "val_loss": val_loss
    }
    st.session_state["history"] = history
    st.session_state["trained"] = True
    return history

# ---------------------------
# Navigation Sidebar
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Analysis", "🧠 Training", "🔎 Prediction"]
)

# ---------------------------
# Pages
# ---------------------------
if page == "🏠 Home":
    st.title("🫁 Tuberculosis Detection from Chest X-rays")
    st.markdown("""
    Welcome to the **Tuberculosis Detection Demo App**.  
    Navigate using the sidebar to explore:
    - **📊 Data Analysis**: Explore synthetic patient dataset.  
    - **🧠 Training**: Simulated model training and evaluation.  
    - **🔎 Prediction**: Upload a chest X-ray and get a demo prediction.  
    """)

elif page == "📊 Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    df = create_sample_data()
    st.subheader("Sample Dataset")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Diagnosis", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Gender vs Diagnosis")
    fig, ax = plt.subplots()
    sns.countplot(x="Gender", hue="Diagnosis", data=df, ax=ax)
    st.pyplot(fig)

elif page == "🧠 Training":
    st.header("🧠 Train Model (Simulated)")
    if st.button("Start Training"):
        history = train_model()
        st.success("✅ Training complete (simulated).")

        # Plot accuracy
        st.subheader("Training History")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history["accuracy"], label="Train Acc")
        ax[0].plot(history["val_accuracy"], label="Val Acc")
        ax[0].set_title("Accuracy")
        ax[0].legend()

        ax[1].plot(history["loss"], label="Train Loss")
        ax[1].plot(history["val_loss"], label="Val Loss")
        ax[1].set_title("Loss")
        ax[1].legend()
        st.pyplot(fig)

        # Simulated classification report + confusion matrix
        st.subheader("Classification Report (Simulated)")
        y_true = np.random.choice([0, 1], 50)
        y_pred = y_true.copy()
        flip_idx = np.random.choice(len(y_true), 5, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]

        report = classification_report(y_true, y_pred, target_names=["Normal", "TB"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix (Simulated)")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "TB"], yticklabels=["Normal", "TB"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

elif page == "🔎 Prediction":
    st.header("🔎 Predict on New X-ray")
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        # Preprocess
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        if st.session_state["trained"]:
            # Simulated prediction
            prob = np.random.rand()
            pred = "TB Detected" if prob > 0.5 else "Normal"
            st.subheader("Prediction Result")
            st.write(f"**{pred}** (Confidence: {prob:.2f})")
        else:
            st.warning("⚠️ Please train the model first (simulated).")

