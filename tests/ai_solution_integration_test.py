import pytest
from unittest import mock
import os
import time

import pytest
import os
from unittest import mock
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
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
        

# Test model training (mocking model saving)
@mock.patch("tensorflow.keras.Model.fit")
@mock.patch("mlflow.log_metric")
def test_train_model_only(mock_log_metric, mock_fit):
    # Mock the fit method
    mock_fit.return_value = mock.MagicMock(
        history={
            'loss': [0.5],
            'accuracy': [0.8],
            'val_loss': [0.6],
            'val_accuracy': [0.75]
        }
    )
    
    # Create a simple mock dataset
    train_ds = tf.data.Dataset.from_tensor_slices(([[1.0]], [[1.0]])).batch(1)
    val_ds = tf.data.Dataset.from_tensor_slices(([[1.0]], [[1.0]])).batch(1)
    test_ds = tf.data.Dataset.from_tensor_slices(([[1.0]], [[1.0]])).batch(1)

    # Create a simple model
    input_shape = (1,)
    model = Sequential([Dense(1, input_shape=input_shape)])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Call the train_and_evaluate_model function
    train_and_evaluate_model(model, train_ds, val_ds, test_ds, 1, "tf_model", "./models/tf_model")
    
    # Assertions
    # Verify model.fit is called
    mock_fit.assert_called_once_with(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=1
    )
    
    # Verify mlflow.log_metric calls
    expected_metrics = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    for metric in expected_metrics:
        assert any(metric in call.args[0] for call in mock_log_metric.call_args_list), \
            f"Metric {metric} was not logged"

# Test model evaluation
def test_model_evaluation():
    dataset = load_data()
    train_ds, val_ds, test_ds = partition_dataset(dataset)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds)
    
    # Create and train model
    input_shape = (256, 256, 3)
    model = create_model(input_shape, 5)
    history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_ds)
    
    # Check that the model accuracy is above a certain threshold (e.g., 60%)
    assert test_accuracy > 0.5, f"Model accuracy is below expected threshold. Got {test_accuracy:.2f}"
    
    # Check that test loss is a finite value
    assert test_loss < float('inf'), f"Test loss is not a finite value: {test_loss}"


