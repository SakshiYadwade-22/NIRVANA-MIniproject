# train.py

from data_loader import load_csv_data, preprocess_csv_data, load_image_data, preprocess_image_data, split_data
from model import build_model
import os

def train_model(csv_data_path, image_dir, saved_model_path):
    # Load and preprocess CSV data
    csv_data = load_csv_data(csv_data_path)
    X_csv, y = preprocess_csv_data(csv_data)
    
    # Load and preprocess image data
    images = load_image_data(image_dir)
    X_images = preprocess_image_data(images)
    
    # Split data into train and test sets
    X_train_csv, X_test_csv, X_train_images, X_test_images, y_train, y_test = split_data(X_csv, X_images, y)
    
    # Build and train the model
    model = build_model(input_shape=X_train_images.shape[1:])
    model.fit(X_train_images, y_train, epochs=10, batch_size=32, validation_data=(X_test_images, y_test))
    
    # Save the model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model.save(saved_model_path)

if __name__ == '__main__':
    csv_data_path = 'data/csv/dyslexia_data.csv'
    image_dir = 'data/images/dyslexia_images'
    saved_model_path = 'saved_models/dyslexia_prediction_model.h5'
    train_model(csv_data_path, image_dir, saved_model_path)
