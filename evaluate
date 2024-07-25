# evaluate.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from data_loader import load_csv_data, preprocess_csv_data, load_image_data, preprocess_image_data, split_data

def evaluate_model(model_path, csv_data_path, image_dir):
    # Load and preprocess CSV data
    csv_data = load_csv_data(csv_data_path)
    X_csv, y = preprocess_csv_data(csv_data)
    
    # Load and preprocess image data
    images = load_image_data(image_dir)
    X_images = preprocess_image_data(images)
    
    # Split data into train and test sets
    _, X_test_csv, _, X_test_images, y_test = split_data(X_csv, X_images, y)
    
    # Load the trained model
    model = load_model(model_path)
    
    # Evaluate the model
    y_pred = model.predict(X_test_images)
    y_pred_classes = np.round(y_pred)
    
    # Print evaluation metrics
    print(confusion_matrix(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes))

if __name__ == '__main__':
    model_path = 'saved_models/dyslexia_prediction_model.h5'
    csv_data_path = 'data/csv/dyslexia_data.csv'
    image_dir = 'data/images/dyslexia_images'
    evaluate_model(model_path, csv_data_path, image_dir)
