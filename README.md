# 🫁 Tuberculosis Detection from Chest X-ray Images

## 📌 Overview
This project detects **Tuberculosis (TB)** from chest X-ray images using **deep learning** and **transfer learning** with models:
- ResNet50
- VGG16
- EfficientNetB0

The goal is to:
1. **Preprocess** X-ray images  
2. **Train** multiple CNN models using transfer learning  
3. **Evaluate & compare** model performance  
4. **Deploy** the best model with a Streamlit web app  

---

## 📂 Project Structure
├── dataset/
│ ├── train/
│ │ ├── NORMAL/
│ │ ├── TUBERCULOSIS/
│ ├── val/
│ │ ├── NORMAL/
│ │ ├── TUBERCULOSIS/
│ ├── test/
│ ├── NORMAL/
│ ├── TUBERCULOSIS/
│
├── train_resnet50.py
├── train_vgg16.py
├── train_efficientnetb0.py
├── app.py
├── README.md

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/tb-detection.git
cd tb-detection
📊 Dataset
You can use:

TB Chest X-ray Dataset from Kaggle:
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Ensure your dataset is structured as shown above.
Each script saves the best model based on validation accuracy:

ResNet50_best.h5

VGG16_best.h5

EfficientNetB0_best.h5
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory("dataset/test", target_size=(224,224), batch_size=32, class_mode='binary')

model = load_model("ResNet50_best.h5")
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc*100:.2f}%")
🌐 Deployment (Streamlit)
Run:

bash
Copy
Edit
streamlit run app.py
Upload an X-ray image and get TB detection results in real-time.
| Model          | Accuracy | Precision | Recall | F1-score |
| -------------- | -------- | --------- | ------ | -------- |
| ResNet50       | xx%      | xx%       | xx%    | xx%      |
| VGG16          | xx%      | xx%       | xx%    | xx%      |
| EfficientNetB0 | xx%      | xx%       | xx%    | xx%      |

---

If you want, I can **extend this README** with:
- Detailed **EDA steps**
- **Sample output images**
- Deployment guide for **Streamlit Cloud or Hugging Face Spaces**  

Do you want me to make that extended README?

