import pandas as pd
import numpy as np
import mlflow
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import boto3
from io import StringIO


# Load data from S3
def load_data():
    bucket_name = os.getenv('BUCKET_NAME')
    file_key = os.getenv('FILE_KEY')

    # Create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    # Read the CSV file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')
    
    # Load into a pandas DataFrame
    return pd.read_csv(StringIO(csv_content))

# Preprocess data
def preprocess_data(df):
    # Select features and target variable
    features = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 'x_wgs84', 'y_wgs84']
    target = 'code_qual'
    
    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning function using GridSearchCV
def train_model_with_grid_search(X_train, y_train):
    # Define the model
    clf = RandomForestClassifier(random_state=42)

    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [10, 20, 30, None],  # Max depth of the trees
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],    # Minimum samples required to be at a leaf node
        'max_features': ['sqrt', 'log2']  # Number of features to consider at each split
    }

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    return grid_search

# Log model and metrics to MLflow
def log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    # Get the best model
    best_clf = grid_search.best_estimator_
    
    # Log hyperparameters using MLflow
    best_params = grid_search.best_params_
    for param, value in best_params.items():
        mlflow.log_param(param, value)  # Log each hyperparameter

    # Make predictions with the best model
    y_pred = best_clf.predict(X_test)

    # Calculate additional metrics
    f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1 score (can be adjusted as needed)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test), multi_class='ovr', average='weighted')  # Multi-class ROC AUC

    # Print Classification Report (can be used for detailed metrics)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Log the metrics to MLflow
    mlflow.log_metric("Train Accuracy", accuracy_score(y_train, best_clf.predict(X_train)))
    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test F1 Score", f1)
    mlflow.log_metric("Test Precision", precision)
    mlflow.log_metric("Test Recall", recall)
    mlflow.log_metric("Test ROC AUC", roc_auc)

    # Log the best model to MLflow
    mlflow.sklearn.log_model(best_clf, artifact_path)

    # Register the model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
    try:
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"Model registered with version: {result.version}")
    except Exception as e:
        print(f"Error registering model: {str(e)}")
 

# Main function to execute the experiment
def run_experiment(experiment_name, artifact_path, registered_model_name):
    start_time = time.time()

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Set experiment's info 
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Ensure no active runs are left open
    if mlflow.active_run():
        mlflow.end_run()

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    # Start a new run
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model with hyperparameter tuning using GridSearchCV
        grid_search = train_model_with_grid_search(X_train, y_train)

        # Log model and metrics
        log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)

    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")


# Entry point for the script
if __name__ == "__main__":
    
    mlflow.set_tracking_uri("https://veeramanicadas-mlflow-server-demo.hf.space")
    experiment_name = "air_quality_tuning"
    mlflow.set_experiment(experiment_name)

    artifact_path = "air_quality_model"
    registered_model_name = "air_quality_best_model"

    run_experiment(experiment_name, artifact_path, registered_model_name)