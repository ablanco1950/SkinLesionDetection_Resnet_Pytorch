# https://www.kaggle.com/code/saadmohamed99/plant-disease-classification

# 1. IMPORT LIBRARIES

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# 2. Extract Data and EDA

DATASET_DIR = "Dir_SkinCancer_Resnet_Pytorch/train"
IMG_SIZE=224

classes = os.listdir(DATASET_DIR)
print("Number of classes:", len(classes))

print(classes)

class_counts = {cls: len(os.listdir(os.path.join(DATASET_DIR, cls))) for cls in classes}

plt.figure(figsize=(10,4))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.xticks(rotation=90)
plt.title("Class Distribution (Imbalanced Dataset)")
plt.show()

def show_sample_images():
    plt.figure(figsize=(12,6))
    for i, cls in enumerate(np.random.choice(classes, 6)):
        img_path = os.path.join(DATASET_DIR, cls, 
                                np.random.choice(os.listdir(os.path.join(DATASET_DIR, cls))))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")
    plt.show()

show_sample_images()

# 3. Image Loading & Cleaning

def load_images(dataset_dir, img_size=IMG_SIZE):
    images = []
    labels = []
    
    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        ContCls=0
        for img_name in os.listdir(cls_path):
            ContCls=ContCls+1
            #if ContCls > 3500: break # to limit number of images
            img_path = os.path.join(cls_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size)) 
                images.append(img)
                labels.append(cls)
            except:
                pass
    
    return np.array(images), np.array(labels)

# 4. Encode Labels & Train/Test Split

X, y = load_images(DATASET_DIR, IMG_SIZE)

print("Images shape:", X.shape)

le = LabelEncoder() # encoder syring classes names into int values
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# 5. Load EfficientNet as Feature Extractor

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

feature_extractor.trainable = False

# 6. Feature Extraction

X_train_pre = preprocess_input(X_train)
X_test_pre = preprocess_input(X_test)

train_features = feature_extractor.predict(X_train_pre, batch_size=32)
test_features = feature_extractor.predict(X_test_pre, batch_size=32)

print("Feature vector shape:", train_features.shape)

# 7. Machine Learning Classifier (SVM)

"""
# Getting the best parameters with GridSearchCV
# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],   # Kernel coefficient
    'kernel': ['rbf', 'poly', 'sigmoid']  # Kernel types
}

# Create the SVM model
svm_model = SVC()

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Configure GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring='accuracy',   # Metric to optimize
    cv=5,                 # 5-fold cross-validation
    verbose=1            # Show progress
   # n_jobs=-1             # Use all CPU cores
)

# Fit the model with grid search
#grid_search.fit(X_train, y_train)
grid_search.fit(train_features, y_train)

# Display best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy: {:.4f}".format(grid_search.best_score_))

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print("\nTest Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# RESULTS
#Best Parameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
#Best Cross-Validation Accuracy: 0.8203
"""

#svm_model = SVC(kernel="rbf", probability=True)
svm_model = SVC(kernel="rbf", C=10, gamma=0.01, probability=True)
#svm_model = SVC(kernel="rbf", C=10, gamma=0.01, probability=False) # probability parameter is not useful in this case
svm_model.fit(train_features, y_train)

y_pred = svm_model.predict(test_features)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

import joblib

joblib.dump(svm_model, "svm_SkinLesions.pkl")
feature_extractor.save("efficientnet_feature_extractor.keras")


