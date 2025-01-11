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
    # Load raw data (assumed to be implemented in load_data function)
    data = load_data()
    
    # Preprocess the data using preprocess_data function (e.g., cleaning, encoding, scaling)
    processed_data = preprocess_data(data)

    # Split the preprocessed data into training and testing sets
    X_train, X_test, y_train, y_test = processed_data

    # Return a dictionary containing raw and processed data for reuse in tests
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
    # Extract training features and target from the preprocessed data
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    
    # Train the model using a GridSearchCV pipeline
    grid_search = train_model_with_grid_search(X_train, y_train)
    
    # Retrieve the best model from GridSearchCV results
    best_model = grid_search.best_estimator_

    # Return the best model and the full grid search object for further testing
    return best_model, grid_search


def test_load_data(preprocessed_data):
    """Test if the load_data function loads the data properly."""
    raw_data = preprocessed_data["raw_data"]
    
    # Verify that the loaded data is a non-empty DataFrame
    assert isinstance(raw_data, pd.DataFrame), "Loaded data is not a DataFrame."
    assert not raw_data.empty, "Loaded data is empty."

    # Check that the target column exists
    assert "code_qual" in raw_data.columns, "Target column 'code_qual' is missing in the raw data."
    
    # Ensure there are no missing values in the data
    assert raw_data.isnull().sum().sum() == 0, "There are missing values in the data."


def test_preprocessing(preprocessed_data):
    """Test if the preprocessing function works as expected."""
    X_train = preprocessed_data["X_train"]
    
    # Ensure training data is non-empty and contains features
    assert X_train.shape[0] > 0, "Training data is empty."
    assert X_train.shape[1] > 0, "Training data has no features."
    
    # Verify that the test set has the same feature structure as the training set
    X_test = preprocessed_data["X_test"]
    assert X_train.shape[1] == X_test.shape[1], "Feature mismatch between training and testing data."


def test_model_training(trained_model, preprocessed_data):
    """Test if the model training works and evaluate the trained model."""
    best_model, grid_search = trained_model
    X_test = preprocessed_data["X_test"]
    y_test = preprocessed_data["y_test"]

    # Verify that the model can make predictions on the test set
    predictions = best_model.predict(X_test)
    assert len(predictions) == len(y_test), "Prediction length does not match test data length."

    # Calculate performance metrics to evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr', average='weighted')

    # Assert that the metrics are within valid ranges
    assert accuracy >= 0, "Accuracy should be non-negative."
    assert 0 <= f1 <= 1, f"F1 score should be between 0 and 1, but got {f1}."
    assert 0 <= precision <= 1, f"Precision should be between 0 and 1, but got {precision}."
    assert 0 <= recall <= 1, f"Recall should be between 0 and 1, but got {recall}."
    assert 0 <= roc_auc <= 1, f"ROC AUC score should be between 0 and 1, but got {roc_auc}."


def test_mlflow_logging(preprocessed_data):
    """Test MLflow logging functionality."""
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]

    # Mock MLflow logging methods to test logging functionality without actual server interaction
    with mock.patch("mlflow.log_metric") as mock_log_metric:
        with mock.patch("mlflow.log_param") as mock_log_param:
            with mock.patch("mlflow.sklearn.log_model") as mock_log_model:
                
                # Start an MLflow run context to test logging calls
                with mlflow.start_run():
                    # Train the model and log relevant metrics and parameters
                    grid_search = train_model_with_grid_search(X_train, y_train)
                    log_model_and_metrics(
                        grid_search, X_train, y_train,
                        preprocessed_data["X_test"], preprocessed_data["y_test"],
                        "random_forest_model", "random_forest_best_model"
                    )
                
                # Verify that MLflow logging methods were called as expected
                assert mock_log_metric.call_count > 0, "MLflow metrics logging was not called."
                assert mock_log_param.call_count > 0, "MLflow parameters logging was not called."
                assert mock_log_model.call_count > 0, "MLflow model logging was not called."

