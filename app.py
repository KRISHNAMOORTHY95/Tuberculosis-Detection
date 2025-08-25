import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import io

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="TB X-ray Classifier", page_icon="🫁", layout="wide")

# ---------------------------
# Paths
# ---------------------------
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "best_model.h5"
HISTORY_PATH = "history.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------------------
# Data Generators
# ---------------------------
def get_datagens():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen, test_gen

# ---------------------------
# Build ResNet50 Model
# ---------------------------
def build_resnet(num_classes):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------
# Save & Load History
# ---------------------------
def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f)

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return None

# ---------------------------
# Pages
# ---------------------------
st.sidebar.title("🫁 TB X-ray Classifier")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Training", "Evaluation", "Prediction"])

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.title("Tuberculosis Detection from Chest X-rays 🫁")
    st.write("""
    This app uses a **ResNet50** deep learning model to classify chest X-rays as **Tuberculosis (TB)** or **Normal**.

    - **EDA** → Explore dataset distribution and sample images.  
    - **Training** → Train the ResNet50 model on your dataset.  
    - **Evaluation** → View accuracy/loss plots, confusion matrix, and classification report.  
    - **Prediction** → Upload an image for TB prediction.  
    """)

# ---------------------------
# EDA Page
# ---------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis 📊")
    if not os.path.exists(TRAIN_DIR):
        st.error("Dataset not found. Please ensure dataset/train, dataset/val, dataset/test exist.")
    else:
        # Class Distribution
        classes = os.listdir(TRAIN_DIR)
        data = []
        for cls in classes:
            data.append({"Diagnosis": cls, "Count": len(os.listdir(os.path.join(TRAIN_DIR, cls)))})
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots()
        sns.barplot(x="Diagnosis", y="Count", data=df, ax=ax)
        st.pyplot(fig)

        # Show Sample Images
        st.subheader("Sample Images")
        for cls in classes:
            st.write(f"Class: {cls}")
            img_files = os.listdir(os.path.join(TRAIN_DIR, cls))[:3]
            cols = st.columns(3)
            for i, img_file in enumerate(img_files):
                img = Image.open(os.path.join(TRAIN_DIR, cls, img_file))
                cols[i].image(img, caption=f"{cls}", use_container_width=True)

# ---------------------------
# Training Page
# ---------------------------
elif page == "Training":
    st.title("Train Model 🏋️")
    if st.button("Start Training"):
        train_gen, val_gen, test_gen = get_datagens()
        model = build_resnet(num_classes=len(train_gen.class_indices))
        
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        mc = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=[es, mc]
        )
        save_history(history)
        st.success("✅ Training complete. Model saved as best_model.h5")

# ---------------------------
# Evaluation Page
# ---------------------------
elif page == "Evaluation":
    st.title("Model Evaluation 📈")
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        _, _, test_gen = get_datagens()

        # Load history
        history = load_history()
        if history:
            st.subheader("Accuracy & Loss Curves")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(history["accuracy"], label="train_acc")
            ax[0].plot(history["val_accuracy"], label="val_acc")
            ax[0].set_title("Accuracy")
            ax[0].legend()
            
            ax[1].plot(history["loss"], label="train_loss")
            ax[1].plot(history["val_loss"], label="val_loss")
            ax[1].set_title("Loss")
            ax[1].legend()
            
            st.pyplot(fig)

        # Confusion Matrix
        st.subheader("Confusion Matrix & Classification Report")
        preds = model.predict(test_gen)
        y_pred = np.argmax(preds, axis=1)
        cm = confusion_matrix(test_gen.classes, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys(), ax=ax)
        st.pyplot(fig)

        report = classification_report(test_gen.classes, y_pred, target_names=test_gen.class_indices.keys(), output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    else:
        st.error("Train the model first!")

# ---------------------------
# Prediction Page
# ---------------------------
elif page == "Prediction":
    st.title("Predict TB from X-ray 🔍")
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_resized = img.resize(IMG_SIZE)
            st.image(img, caption="Uploaded X-ray", use_container_width=True)
            
            arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)
            pred = model.predict(arr)
            classes = list(model.layers[-1].output_shape[-1] for _ in range(len(pred[0])))
            class_labels = list(model.class_names) if hasattr(model, 'class_names') else ["Class0", "Class1"]

            result = np.argmax(pred)
            st.success(f"Prediction: **{class_labels[result]}** (Confidence: {pred[0][result]:.2f})")
    else:
        st.error("Train the model first!")

