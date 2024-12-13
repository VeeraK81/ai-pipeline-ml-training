    # import pytest
    # from unittest import mock
    # from app.air_quality_ml_train import load_data, preprocess_data, forecast_pollutant_levels, log_model_and_metrics
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    # from sklearn.model_selection import GridSearchCV
    # import pandas as pd
    # import numpy as np


    # data = load_data()
    # X_train, X_test, y_train, y_test = preprocess_data(data)
    # clf = RandomForestClassifier(random_state=42)
    # clf.fit(X_train, y_train)

    # param_grid = {
    #     'n_estimators': [10],  # Simplified for test speed
    #     'max_depth': [10],
    #     'min_samples_split': [2],
    #     'min_samples_leaf': [1]
    # }
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2)
    # grid_search.fit(X_train, y_train)

    # def test_load_data():
    #     """Test if the load_data function loads the data properly."""
        

    #     assert isinstance(data, pd.DataFrame), "Loaded data is not a DataFrame."
    #     expected_columns = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 'x_wgs84', 'y_wgs84', 'code_qual']
    #     assert all(col in data.columns for col in expected_columns), "Missing expected columns in loaded data."
    #     assert len(data) > 0, "Loaded data is empty."


    # def test_preprocess_data():
    #     """Test the preprocess_data function."""

    #     assert len(X_train) > 0 and len(X_test) > 0, "Train or test features are empty."
    #     assert len(y_train) > 0 and len(y_test) > 0, "Train or test labels are empty."
    #     assert X_train.shape[1] == 7, "Number of features in training set is incorrect."


    # def test_forecast_pollutant_levels():
    #     """Test the forecast_pollutant_levels function."""
    #     pollutants = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25']
    #     predictions = forecast_pollutant_levels(data, pollutants)

    #     assert isinstance(predictions, dict), "Forecast results are not a dictionary."
    #     assert all(pollutant in predictions for pollutant in pollutants), "Missing pollutant in forecast results."
    #     for forecast in predictions.values():
    #         assert forecast is not None, "Forecast for a pollutant is None."
    #         assert len(forecast) == 2, "Forecast does not have two steps."


    # def test_log_model_and_metrics():
    #     """Test the log_model_and_metrics function."""

    #     # Mock MLflow logging
    #     with mock.patch("mlflow.log_metric") as mock_log_metric:
    #         log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, "artifact_path", "model_name")

    #         assert mock_log_metric.call_count > 0, "MLflow metrics logging was not called."


    # def test_metrics():
        
    #     best_clf = grid_search.best_estimator_
    #     y_pred = best_clf.predict(X_test)

    #     f1 = f1_score(y_test, y_pred, average='weighted')
    #     precision = precision_score(y_test, y_pred, average='weighted')
    #     recall = recall_score(y_test, y_pred, average='weighted')
    #     roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test), multi_class='ovr', average='weighted')

    #     assert f1 > 0, f"F1 Score is too low: {f1}"
    #     assert precision > 0, f"Precision is too low: {precision}"
    #     assert recall > 0, f"Recall is too low: {recall}"
    #     assert roc_auc > 0, f"ROC AUC is too low: {roc_auc}"
    
    
import pytest
from unittest import mock
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from app.air_quality_ml_train import load_data, preprocessing, create_pipeline, train_model


@pytest.fixture
def preprocessed_data():
    """Fixture to load and preprocess data for testing."""
    data = load_data()
    processed_data = preprocessing(data)

    # Define features and target
    features = [col for col in processed_data.columns if col not in ['valeur', 'date_debut']]
    X = processed_data[features]
    y = processed_data['valeur']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return {
        "raw_data": data,
        "processed_data": processed_data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

@pytest.fixture
def trained_pipeline(preprocessed_data):
    """Fixture to create and train a pipeline for testing."""
    pipeline = create_pipeline()
    pipeline.fit(preprocessed_data["X_train"], preprocessed_data["y_train"])
    return pipeline

def test_load_data(preprocessed_data):
    """Test if the load_data function loads the data properly."""
    raw_data = preprocessed_data["raw_data"]
    assert isinstance(raw_data, pd.DataFrame), "Loaded data is not a DataFrame."
    assert not raw_data.empty, "Loaded data is empty."
    assert "valeur" in raw_data.columns, "Target column 'valeur' is missing in the raw data."

def test_preprocessing(preprocessed_data):
    """Test if the preprocessing function works as expected."""
    processed_data = preprocessed_data["processed_data"]
    expected_features = ['hour', 'day_of_week', 'month']
    
    for feature in expected_features:
        assert feature in processed_data.columns, f"Feature '{feature}' is missing after preprocessing."

    assert 'valeur' in processed_data.columns, "Target column 'valeur' is missing in the processed data."

def test_pipeline_training(trained_pipeline, preprocessed_data):
    """Test if the pipeline trains successfully."""
    pipeline = trained_pipeline
    X_test = preprocessed_data["X_test"]
    y_test = preprocessed_data["y_test"]

    # Ensure predictions work
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(y_test), "Pipeline prediction length mismatch with test data."

    # Calculate performance metrics to ensure the model is making predictions
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Basic assertions on performance metrics (these can be adjusted based on expected outcomes)
    assert mse >= 0, "Mean Squared Error should be non-negative."
    assert mae >= 0, "Mean Absolute Error should be non-negative."
    assert 0 <= r2 <= 1, f"R2 score should be between 0 and 1, but got {r2}."

def test_mlflow_logging(preprocessed_data):
    """Test MLflow logging functionality."""
    # Unpack the preprocessed data fixture
    processed_data = preprocessed_data["processed_data"]
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    
    # Mock MLflow logging
    with mock.patch("mlflow.log_metric") as mock_log_metric:
        with mock.patch("mlflow.log_param") as mock_log_param:
            # Call the train_model function with the processed data
            train_model(processed_data, "air_quality_tuning")
            
            # Assert that MLflow logging was called
            assert mock_log_metric.call_count > 0, "MLflow metrics logging was not called."
            assert mock_log_param.call_count > 0, "MLflow parameters logging was not called."
