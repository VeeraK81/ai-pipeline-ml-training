import pytest
from unittest import mock
import os
import time

import pytest
import os
from unittest import mock
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.ai_solution_ml_train import load_data, preprocess_data, create_model, train_and_evaluate_model, partition_dataset
import mlflow
from mlflow import log_metric, log_artifact

# Test data loading
def test_load_data():
    dataset = load_data()
    # Check that dataset is not empty
    assert len(dataset) > 0, "Dataset is empty"


# Test data preprocessing
def test_preprocess_data():
    dataset = load_data()
    train_ds, val_ds, test_ds = partition_dataset(dataset)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds)
    
    # Check that train, validation, and test datasets are not empty
    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"
    assert len(test_ds) > 0, "Test dataset is empty"

# Test model creation
def test_create_model():
    input_shape = (256, 256, 3)
    n_classes = 5
    model = create_model(input_shape, n_classes)
    
    # Check that model is an instance of Sequential
    assert isinstance(model, tf.keras.models.Sequential), "Model is not a Sequential model"
    
    # Check that the model has the expected layers
    layer_names = [layer.name for layer in model.layers]
    assert any("conv2d" in layer_name for layer_name in layer_names), "Conv2D layer missing in model"
    assert any("dense" in layer_name for layer_name in layer_names), "Dense layer missing in model"
        

# Define a custom save function to bypass actual model saving
def dummy_save(filepath, *args, **kwargs):
    print(f"Model save function called with path: {filepath}")

# Define a dummy logging function for metrics
def dummy_log_metric(metric_name, value, *args, **kwargs):
    print(f"Metric logged: {metric_name} = {value}")

# Define a dummy logging function for artifacts
def dummy_log_artifact(artifact_path, *args, **kwargs):
    print(f"Artifact logged: {artifact_path}")

def test_train_and_evaluate_model():
    # Patch save and logging methods by overriding them
    original_save = tf.keras.Model.save
    original_log_metric = mlflow.log_metric
    original_log_artifact = mlflow.log_artifact

    try:
        # Replace the original methods with dummy ones
        tf.keras.Model.save = dummy_save
        mlflow.log_metric = dummy_log_metric
        mlflow.log_artifact = dummy_log_artifact

        # Load and preprocess data
        dataset = load_data()
        train_ds, val_ds, test_ds = partition_dataset(dataset)
        train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds)

        # Create model
        input_shape = (256, 256, 3)
        model = create_model(input_shape, 5)

        # Train and evaluate the model
        train_and_evaluate_model(model, train_ds, val_ds, test_ds, 5, "tf_model", "./models/tf_model")

        # Validation of expected calls
        print("Test passed: Save and logging functions were successfully replaced.")
    finally:
        # Restore the original methods
        tf.keras.Model.save = original_save
        mlflow.log_metric = original_log_metric
        mlflow.log_artifact = original_log_artifact

# Test model evaluation
def test_model_evaluation():
    dataset = load_data()
    train_ds, val_ds, test_ds = partition_dataset(dataset)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds)
    
    # Create and train model
    input_shape = (256, 256, 3)
    model = create_model(input_shape, 5)
    history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_ds)
    
    # Check that the model accuracy is above a certain threshold (e.g., 60%)
    assert test_accuracy > 0.6, f"Model accuracy is below expected threshold. Got {test_accuracy:.2f}"
    
    # Check that test loss is a finite value
    assert test_loss < float('inf'), f"Test loss is not a finite value: {test_loss}"


