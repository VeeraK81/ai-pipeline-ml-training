# import pytest
# from unittest import mock
# from app.fraud_detection_train import load_data, preprocess_data, create_pipeline, train_model
# from io import StringIO
# import os


# # Test data loading
# def test_load_data():
#     # S3 Bucket and Key
#     df = load_data()
 
#     assert not df.empty, "Dataframe is empty"

# # Test data preprocessing
# def test_preprocess_data():
#     df = load_data()
#     X_train, X_test, y_train, y_test = preprocess_data(df)
#     assert len(X_train) > 0, "Training data is empty"
#     assert len(X_test) > 0, "Test data is empty"

# # Test pipeline creation
# def test_create_pipeline():
#     pipe = create_pipeline()
#     assert "scaler" in pipe.named_steps, "Scaler missing in pipeline"
#     assert "classifier" in pipe.named_steps, "Classifier missing in pipeline"

# # Test model training (mocking GridSearchCV)
# # @mock.patch('app.train.GridSearchCV.fit', return_value=None)
# @mock.patch('app.fraud_detection_train.GridSearchCV.fit', return_value=None)
# def test_train_model(mock_fit):
#     pipe = create_pipeline()
#     X_train, X_test, y_train, y_test = preprocess_data(load_data())
#     param_grid = {
#         "classifier__n_estimators": [100, 150],
#         "classifier__max_depth": [10, 20, 30],
#         "classifier__min_samples_split": [2, 5],
#         "classifier__min_samples_leaf": [1, 2]
#     }   
#     model = train_model(pipe, X_train, y_train, param_grid)
#     assert model is not None, "Model training failed"

import pytest
from unittest import mock
from app.fraud_detection_train import load_data, preprocess_data, create_pipeline, train_model
from io import StringIO
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Test data loading
def test_load_data():
    # S3 Bucket and Key
    df = load_data()
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

def test_create_pipeline():
    pipelines = create_pipeline()
    
    # Check that the returned object is a list of tuples
    assert isinstance(pipelines, list), "Expected a list of pipelines"  
    
    for name, pipe in pipelines:
        # Check that each entry is a tuple with a name and a Pipeline
        assert isinstance(name, str), "Pipeline name should be a string"
        assert isinstance(pipe, Pipeline), f"Expected a Pipeline object, got {type(pipe)}"
        
        # Check that the pipeline contains the 'scaler' and 'classifier' steps
        assert "scaler" in pipe.named_steps, f"Scaler missing in pipeline '{name}'"
        assert "classifier" in pipe.named_steps, f"Classifier missing in pipeline '{name}'"

        # Optional: check that the classifier type matches the expected one
        if name == "RandomForest":
            assert isinstance(pipe.named_steps["classifier"], RandomForestClassifier), "Expected RandomForestClassifier"
        elif name == "LogisticRegression":
            assert isinstance(pipe.named_steps["classifier"], LogisticRegression), "Expected LogisticRegression"


# Test model training (mocking GridSearchCV)
def test_train_model():
    # Load data directly
    X_train, X_test, y_train, y_test = preprocess_data(load_data())
    
    # Define the parameter grid for both models
    param_grids = {
        "RandomForest": {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        },
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10],  # Regularization strength
            "classifier__penalty": ['l2'],   # Regularization type
            "classifier__solver": ['lbfgs', 'liblinear']  # Solvers for optimization
        }
    }

    # Test each pipeline
    pipelines = create_pipeline()
    
    for name, pipe in pipelines:
        print(f"Training with pipeline: {name}")
        
        # Train model with GridSearchCV and parameter grid
        model = train_model([(name, pipe)], X_train, y_train, {name: param_grids[name]})
        
        # Ensure the model is returned
        assert model is not None, f"Model training failed for {name}"

        # Check the best model selection
        assert hasattr(model, 'best_score_'), f"Model for {name} does not have a 'best_score_' attribute."
        assert hasattr(model, 'best_params_'), f"Model for {name} does not have 'best_params_' attribute."

