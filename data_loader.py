# data_loader.py

import pandas as pd
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_csv_data(file_path):
    data = pd.read_csv(file_path)  # Adjust based on your CSV data format
    return data

def preprocess_csv_data(data):
    X = data.drop('target', axis=1)  # Adjust 'target' to your target variable
    y = data['target']
    
    # Example: Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def load_image_data(image_dir):
    image_files = os.listdir(image_dir)
    images = []
    for file in image_files:
        image = imread(os.path.join(image_dir, file), as_gray=True)  # Load image in grayscale
        image_resized = resize(image, (64, 64))  # Resize image to a consistent size
        images.append(image_resized)
    return np.array(images)

def preprocess_image_data(images):
    # Normalize pixel values (assuming grayscale images)
    images_normalized = images / 255.0
    return images_normalized

def split_data(X_csv, X_images, y, test_size=0.2):
    X_train_csv, X_test_csv, X_train_images, X_test_images, y_train, y_test = train_test_split(
        X_csv, X_images, y, test_size=test_size, random_state=42)
    return X_train_csv, X_test_csv, X_train_images, X_test_images, y_train, y_test
