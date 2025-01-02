import pytest
from unittest import mock
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow
from app.air_quality_ml_train import load_data, preprocess_data, train_model_with_grid_search, log_model_and_metrics


@pytest.fixture
def preprocessed_data():
    """Fixture to load and preprocess data for testing."""
    data = load_data()  # Load data from CSV (assuming this is already done)
    processed_data = preprocess_data(data)

    # Define features and target
    X_train, X_test, y_train, y_test = processed_data

    return {
        "raw_data": data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

@pytest.fixture
def trained_model(preprocessed_data):
    """Fixture to train model using GridSearchCV."""
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    
    # Train model with GridSearchCV
    grid_search = train_model_with_grid_search(X_train, y_train)
    
    # Get the best estimator from GridSearchCV
    best_model = grid_search.best_estimator_

    return best_model, grid_search


def test_load_data(preprocessed_data):
    """Test if the load_data function loads the data properly."""
    raw_data = preprocessed_data["raw_data"]
    assert isinstance(raw_data, pd.DataFrame), "Loaded data is not a DataFrame."
    assert not raw_data.empty, "Loaded data is empty."
    assert "code_qual" in raw_data.columns, "Target column 'code_qual' is missing in the raw data."
    assert raw_data.isnull().sum().sum() == 0, "There are missing values in the data."


def test_preprocessing(preprocessed_data):
    """Test if the preprocessing function works as expected."""
    X_train = preprocessed_data["X_train"]
    
    assert X_train.shape[0] > 0, "Training data is empty."
    assert X_train.shape[1] > 0, "Training data has no features."
    
    # Ensure that the test set has the same feature columns as the training set
    X_test = preprocessed_data["X_test"]
    assert X_train.shape[1] == X_test.shape[1], "Feature mismatch between training and testing data."


def test_model_training(trained_model, preprocessed_data):
    """Test if the model training works and evaluate the trained model."""
    best_model, grid_search = trained_model
    X_test = preprocessed_data["X_test"]
    y_test = preprocessed_data["y_test"]

    # Ensure predictions work
    predictions = best_model.predict(X_test)
    assert len(predictions) == len(y_test), "Prediction length does not match test data length."

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr', average='weighted')

    # Basic assertions on metrics (can adjust thresholds as needed)
    assert accuracy >= 0, "Accuracy should be non-negative."
    assert 0 <= f1 <= 1, f"F1 score should be between 0 and 1, but got {f1}."
    assert 0 <= precision <= 1, f"Precision should be between 0 and 1, but got {precision}."
    assert 0 <= recall <= 1, f"Recall should be between 0 and 1, but got {recall}."
    assert 0 <= roc_auc <= 1, f"ROC AUC score should be between 0 and 1, but got {roc_auc}."


def test_mlflow_logging(preprocessed_data):
    """Test MLflow logging functionality."""
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]

    # Mock MLflow logging
    with mock.patch("mlflow.log_metric") as mock_log_metric:
        with mock.patch("mlflow.log_param") as mock_log_param:
            with mock.patch("mlflow.sklearn.log_model") as mock_log_model:
                
                # Start a new run within the context
                with mlflow.start_run():
                    # Call the train_model_with_grid_search function with preprocessed data
                    grid_search = train_model_with_grid_search(X_train, y_train)
                    
                    # Log the model and metrics
                    log_model_and_metrics(grid_search, X_train, y_train, preprocessed_data["X_test"], preprocessed_data["y_test"], "random_forest_model", "random_forest_best_model")
                
                # Assert that MLflow logging methods were called
                assert mock_log_metric.call_count > 0, "MLflow metrics logging was not called."
                assert mock_log_param.call_count > 0, "MLflow parameters logging was not called."
                assert mock_log_model.call_count > 0, "MLflow model logging was not called."
