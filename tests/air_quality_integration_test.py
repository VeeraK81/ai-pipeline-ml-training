import pytest
from unittest import mock
from app.air_quality_ml_train import load_data, preprocess_data, train_model_with_grid_search, log_model_and_metrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd




data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)
grid_search = train_model_with_grid_search(X_train, y_train)


# Test to check if the data loading function works properly
def test_load_data():
    
    loaded_df = data
    # Check if the data loaded is a DataFrame
    assert isinstance(loaded_df, pd.DataFrame), "Data is not loaded as DataFrame."

    # Define the expected columns
    expected_columns = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 'x_wgs84', 'y_wgs84', 'code_qual']

    # Check if all expected columns are present in the loaded data
    for column in expected_columns:
        assert column in loaded_df.columns, f"Column '{column}' is missing in the loaded data."


    assert loaded_df.shape[0] > 0, "Number of rows in the loaded data is incorrect."
    
    
# Test to check if the preprocessing step works
def test_preprocess_data():

    # Check if the split produces correct train-test sizes
    assert len(X_train) > 0, "Training data is empty"
    assert len(y_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"
    assert len(y_test) > 0, "Test data is empty"
    

# Test GridSearchCV model training (using small data for fast testing)
def test_train_model_with_grid_search():
        
    # Check that grid search fits the model
    assert grid_search.best_estimator_ is not None, "No best model found after GridSearchCV."
    
# Test if metrics (accuracy, f1, precision, recall, ROC AUC) are calculated correctly
def test_metrics():
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test), multi_class='ovr', average='weighted')

    assert f1 > 0, f"F1 Score is too low: {f1}"
    assert precision > 0, f"Precision is too low: {precision}"
    assert recall > 0, f"Recall is too low: {recall}"
    assert roc_auc > 0, f"ROC AUC is too low: {roc_auc}"

