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

DATASET_DIR = "Dir_Test_SkinCancer_Resnet_Pytorch/test"
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
    img_names=[]
    
    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size)) 
                images.append(img)
                labels.append(cls)
                img_names.append(img_name)
            except:
                pass
    
    return np.array(images), np.array(labels), np.array(img_names)

# 4. Encode Labels & Train/Test Split

X_test, y_test, img_names = load_images(DATASET_DIR, IMG_SIZE)

print("Images shape:", X_test.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y_test)

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#)


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

#X_train_pre = preprocess_input(X_train)
X_test_pre = preprocess_input(X_test)

#train_features = feature_extractor.predict(X_train_pre, batch_size=32)
test_features = feature_extractor.predict(X_test_pre, batch_size=32)

print("Feature vector shape:", test_features.shape)

# 7. Machine Learning Classifier (SVM)

svm_model = SVC(kernel="rbf", probability=True)
#svm_model.fit(train_features, y_train)
import joblib
svm_model=joblib.load("svm_SkinLesions.pkl")

y_pred = svm_model.predict(test_features)

#print(y_pred)

ContError=0
ContHit=0
for i in range(len(y_pred)):
    if classes[y_pred[i]] != y_test[i]:
       print( "ERROR in image " + img_names[i] + " is assigned " + classes[y_pred[i]] + " true es " + y_test[i])
       ContError=ContError+1
    else:
         ContHit=ContHit+1
print("")
print("Errors " + str(ContError))
print("Hits " + str(ContHit))
print("Accuracy:" + str(ContHit/(ContHit+ContError)))
print("")
print("")

print("Accuracy:", accuracy_score(y_encoded, y_pred))
print(classification_report(y_encoded, y_pred, target_names=le.classes_))


