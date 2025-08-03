#!/usr/bin/env python
# coding: utf-8

# In[103]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report


# **Load Data**

# In[9]:


pip install kaggle


# In[41]:


import os
os.chdir(r"D:\DS projects\TB")


# **load the datset**

# In[44]:


#create new dataset and mpbve 20 per to train datset
import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = Path("data/train")
test_dir = Path("data/test")
test_ratio = 0.1  # 10% for testing

# Create test directory
for class_name in ["TB", "Normal"]:
    os.makedirs(test_dir / class_name, exist_ok=True)

    # Get all image file paths in each class folder
    class_dir = source_dir / class_name
    all_images = list(class_dir.glob("*.*"))
    
    # Shuffle and split
    random.shuffle(all_images)
    num_test = int(len(all_images) * test_ratio)
    test_images = all_images[:num_test]

    # Move images to test directory
    for img_path in test_images:
        dest_path = test_dir / class_name / img_path.name
        shutil.move(str(img_path), str(dest_path))

print("✅ Test split completed. Images moved to 'data/test'")


# In[46]:


# Paths
source_dir = Path("data/train")
val_dir = Path("data/val")
val_ratio = 0.2  # 20% for testing

# Create test directory
for class_name in ["TB", "Normal"]:
    os.makedirs(val_dir / class_name, exist_ok=True)

    # Get all image file paths in each class folder
    class_dir = source_dir / class_name
    all_images = list(class_dir.glob("*.*"))
    
    # Shuffle and split
    random.shuffle(all_images)
    num_val = int(len(all_images) * val_ratio)
    val_images = all_images[:num_val]

    # Move images to test directory
    for img_path in val_images:
        dest_path = val_dir / class_name / img_path.name
        shutil.move(str(img_path), str(dest_path))

print("✅ Test split completed. Images moved to 'data/val'")


# **Data cleaning and preprocessing**

# In[52]:


img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
    )

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    './data/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
    )

val_data = datagen.flow_from_directory(
    './data/val',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)


# In[54]:


#Handle Class Imblanace: Use class weights to pay more attention to the minority class during training.


# In[56]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define the class labels (0 for Normal, 1 for TB)
class_labels = ['Normal', 'TB']

class_counts = [371, 1796]

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique([0, 1]),
    y=np.array([0]*371 + [1]*1796)
)

class_weight_dict = dict(zip([0, 1], class_weights))
print("Class weights:", class_weight_dict)


# **to handle missing imahe**

# In[59]:


from PIL import Image
import os

def check_image_validity(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except (IOError, SyntaxError):
        print(f"Corrupt image: {image_path}")
        return False

image_folder = 'data/train/TB'

# Loop through all images and check validity
valid_images = []
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if check_image_validity(image_path):
        valid_images.append(image_path)

print(f"Valid images: {len(valid_images)}")


# In[61]:


def check_image_validity(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except (IOError, SyntaxError):
        print(f"Corrupt image: {image_path}")
        return False

image_folder = 'data/train/Normal'

# Loop through all images and check validity
valid_images = []
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if check_image_validity(image_path):
        valid_images.append(image_path)

print(f"Valid images: {len(valid_images)}")


# **Step :2 Exploratory data analysis**

# In[68]:


def plot_class_distribution(directory):
    class_counts = {}
    
    # Loop through subdirectories (class names)
    for class_name in os.listdir(directory):
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            class_counts[class_name] = len(os.listdir(class_folder))
    
    # Plot bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color=['darkblue', 'gray'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

# Example usage:
plot_class_distribution('data/train')


# In[74]:


get_ipython().system('pip install opencv-python')


# In[80]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def get_pixel_statistics(directory, target_size=(224, 224)):
    pixel_means = []
    pixel_stds = []

    for class_name in os.listdir(directory):
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = img.astype('float32') / 255.0
                    pixel_means.append(np.mean(img))
                    pixel_stds.append(np.std(img))

    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.suptitle('📊 Chest X-Ray Pixel Intensity Analysis', fontsize=16, fontweight='bold')

    plt.subplot(1, 2, 1)
    plt.hist(pixel_means, bins=40, color='skyblue', edgecolor='black')
    plt.title('Mean Pixel Intensity')
    plt.xlabel('Mean Value')
    plt.ylabel('Image Count')

    plt.subplot(1, 2, 2)
    plt.hist(pixel_stds, bins=40, color='salmon', edgecolor='black')
    plt.title('Standard Deviation of Pixel Intensity')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Image Count')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit main title
    plt.show()

# Example usage:
get_pixel_statistics('data/train')


# In[82]:


import matplotlib.image as mpimg
import random

def display_sample_images(directory, num_samples=5):
    class_names = os.listdir(directory)
    
    plt.figure(figsize=(15, 10))

    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            # Pick a few random images from this class
            image_files = os.listdir(class_folder)
            random_images = random.sample(image_files, num_samples)

            for i, image_file in enumerate(random_images):
                img_path = os.path.join(class_folder, image_file)
                img = mpimg.imread(img_path)
                
                plt.subplot(len(class_names), num_samples, idx * num_samples + i + 1)
                plt.imshow(img)
                plt.axis('off')
                if i == 0:
                    plt.title(class_name)
    
    plt.tight_layout()
    plt.show()

# Example usage:
display_sample_images('data/train')


# **Model development**

# In[85]:


base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) 
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[87]:


model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[89]:


model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    class_weight=class_weight_dict)


# In[91]:


model.fit(
    train_data,
    validation_data=val_data,
    epochs=5)
# WITHOUT CLASS WEIGHTS


# In[93]:


# Performance Metrics on Test Dataset:

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)


# In[95]:


y_true = test_data.classes

# Get predicted probabilities
y_pred_probs = model.predict(test_data)


# In[99]:


# Convert to binary predictions
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
# Class names
class_labels = list(test_data.class_indices.keys())


# In[105]:


print(classification_report(y_true, y_pred, target_names=class_labels))


# In[107]:


# Model has performed poorly, as it is unable to identify even a single "Normal" Xray. This is due to class imbalance, 
# using class weights didnt approve accuracy, 
# hence we can increase the samples in Normal class using Oversampling.


# In[109]:


from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

normal_path = 'data/train/Normal'
tb_path = 'data/train/TB'

normal_images = os.listdir(normal_path)
tb_images = os.listdir(tb_path)

print(f"Original Normal images: {len(normal_images)}")
print(f"TB images: {len(tb_images)}")

# Oversample by creating augmented copies of "Normal" images
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

output_dir = normal_path 
n_to_generate = len(tb_images) - len(normal_images)

print(f"Generating {n_to_generate} augmented Normal images...")

i = 0
while i < n_to_generate:
    img_name = random.choice(normal_images)
    img_path = os.path.join(normal_path, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1,) + image.shape)

    for batch in aug.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i >= n_to_generate:
            break

print("Augmentation done.")


# In[110]:


# Reinitialize the train and validation data set

img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    './data/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
    )

val_data = datagen.flow_from_directory(
    './data/val',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)


# In[113]:


def build_model(hp):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Define input
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Pass through base model
    x = base_model(inputs, training=False)
    
    # Add dropout layers with tunable rate
    x = layers.GlobalAveragePooling2D()(x)
    
    # Hyperparameter: Tune dropout rate between 0.3 and 0.6
    x = layers.Dropout(hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1))(x)

    # Add Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # Define model
    resmodel = models.Model(inputs, x)

    # Compile model with tunable learning rate
    resmodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# In[121]:


get_ipython().system('pip install keras-tuner --upgrade')


# In[125]:


import keras_tuner as kt


# In[127]:


# Initialize the tuner

tuner = kt.Hyperband(
    build_model,  # The model-building function
    objective='val_accuracy',  # Metric to optimize
    max_epochs=3,  # Max epochs for each trial
    hyperband_iterations=2,  # Number of Hyperband iterations
    directory='hyperparameter_tuning',  # Directory to store results
    project_name='tb_detection_tuning'  # Project name
)

# Search for the best hyperparameters
tuner.search(train_data, epochs=3, validation_data=val_data)


# In[129]:


# Get the best model
res_best_model = tuner.get_best_models(num_models=1)[0]

# Get the best hyperparameters
res_best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best dropout rate: {res_best_hp.get('dropout')}")
print(f"Best learning rate: {res_best_hp.get('learning_rate')}")


# In[131]:


res_best_model.fit(train_data, epochs=10, validation_data=val_data)


# In[137]:


# Classification report, ROC_AUC:

y_true = test_data.classes
# Get predicted probabilities
y_pred_probs = res_best_model.predict(test_data)
# Convert to binary predictions
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
# Class names
class_labels = list(test_data.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))
print("ROC-AUC Score:", roc_auc_score(y_true, y_pred_probs))


# In[139]:


# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# Final model
vgg_model = models.Model(inputs, outputs)
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[141]:


vgg_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


# In[143]:


# Classification report, ROC_AUC:

y_true = test_data.classes
# Get predicted probabilities
y_pred_probs = vgg_model.predict(test_data)
# Convert to binary predictions
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
# Class names
class_labels = list(test_data.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))
print("ROC-AUC Score:", roc_auc_score(y_true, y_pred_probs))


# In[145]:


# Load base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Build model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

eff_model = models.Model(inputs, outputs)

# Compile model
eff_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[147]:


eff_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


# In[148]:


# Classification report, ROC_AUC:

y_true = test_data.classes
# Get predicted probabilities
y_pred_probs = eff_model.predict(test_data)
# Convert to binary predictions
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
# Class names
class_labels = list(test_data.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))
print("ROC-AUC Score:", roc_auc_score(y_true, y_pred_probs))


# In[151]:


# Save the model
vgg_model.save('tb_detection_model_.keras')


# In[ ]:




