import pandas as pd
import numpy as np
import mlflow
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import boto3
import os
from io import StringIO

    
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from tensorflow.keras import layers, models
from datetime import datetime
import zipfile


# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 5
CHANNELS = 3
N_CLASSES = 5
EXPERIMENT_NAME = "plant_village_image_classification"
MODEL_DIR = "./models/tf_model"
ARTIFACT_PATH = "tf_model"

# Enable MLflow autologging for TensorFlow
mlflow.tensorflow.autolog()

# Load dataset
def load_data():
    # bucket_name = os.getenv('BUCKET_NAME')
    # file_key = os.getenv('FILE_KEY')

    # S3 client setup
    s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    bucket_name = "flow-bucket-ml"
    file_key = "ai-pipeline-solution/plant_village_dataset/Potato_Disease.zip"  # File key in S3
    local_file_path = "./plant_village_dataset/Potato_Disease.zip"  # Local path to save the file
    extract_dir = "./plant_village_dataset/Potato_Disease"  # Directory to extract files
    
    if os.path.exists(local_file_path):
        os.remove(local_file_path)

    try:
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the dataset
        print("Downloading dataset...")
        s3_client.download_file(bucket_name, file_key, local_file_path)

        # Extract the dataset if it's a zip file
        print("Extracting dataset...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Load the dataset
        print("Loading dataset...")
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            extract_dir,
            shuffle=True,
            image_size=(256, 256),  # Replace with IMAGE_SIZE constant
            batch_size=32  # Replace with BATCH_SIZE constant
        )
        print("Dataset loaded successfully!")

    except Exception as e:
        print(f"Error: {e}")
    return dataset

# Partition dataset
def partition_dataset(ds, train_split=0.8, val_split=0.1, shuffle_size=10000):
    ds_size = len(ds)
    ds = ds.shuffle(shuffle_size, seed=42)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)
    return train_ds, val_ds, test_ds

# Data preprocessing
def preprocess_data(train_ds, val_ds, test_ds):
    # Data augmentation
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Optimize dataset loading
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds

# Define the CNN model
def create_model(input_shape, n_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# Train and evaluate model
def train_and_evaluate_model(model, train_ds, val_ds, test_ds, epochs, artifact_path, model_dir):
    with mlflow.start_run():
        # Train the model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )

        # Log training and validation metrics
        train_loss = history.history['loss']
        train_accuracy = history.history['accuracy']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']

        for epoch in range(epochs):
            mlflow.log_metric("train_loss", train_loss[epoch], step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy[epoch], step=epoch)
            mlflow.log_metric("val_loss", val_loss[epoch], step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy[epoch], step=epoch)

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_ds)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Save the model
        os.makedirs(model_dir, exist_ok=True)
        model.save(f"{model_dir}/model.keras")
        mlflow.log_artifact(f"{model_dir}/model.keras", artifact_path=artifact_path)



# Main function
def run_experiment():
    mlflow.set_tracking_uri("http://localhost:8082")
    
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and preprocess data
    dataset = load_data()
    train_ds, val_ds, test_ds = partition_dataset(dataset)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds)

    # Create and train model
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    model = create_model(input_shape, N_CLASSES)
    train_and_evaluate_model(model, train_ds, val_ds, test_ds, EPOCHS, ARTIFACT_PATH, MODEL_DIR)


if __name__ == "__main__":

    run_experiment()


