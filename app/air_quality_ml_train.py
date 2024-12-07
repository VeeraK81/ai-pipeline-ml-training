import pandas as pd
import numpy as np
import mlflow
import time
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import os


# Sample data loading function from S3 (this can be replaced with file reading)
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


# Preprocess data function
def preprocess_data(df):
    features = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 'x_wgs84', 'y_wgs84']
    target = 'code_qual'
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ARIMA model fitting and forecasting
def forecast_pollutant_levels(df, pollutants):
    predictions = {}
    for pollutant in pollutants:
        if df[pollutant].nunique() == 1:
            print(f"Handling constant values for {pollutant}. Forecasted values will be the last observed value.")
            predictions[pollutant] = np.full(2, df[pollutant].iloc[-1])
            print(f"Forecasted {pollutant} levels for the next 2 hours: {predictions[pollutant]}")
            continue
        
        print(f"Fitting ARIMA model for {pollutant}...")
        try:
            model = ARIMA(df[pollutant], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=2)
            predictions[pollutant] = forecast
            print(f"Forecasted {pollutant} levels for the next 2 hours: {forecast}")
        except Exception as e:
            print(f"Error fitting ARIMA model for {pollutant}: {e}")
            predictions[pollutant] = None

    return predictions


# Function to log the model and metrics to MLflow
def log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test), multi_class='ovr', average='weighted')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    mlflow.log_metric("Train Accuracy", accuracy_score(y_train, best_clf.predict(X_train)))
    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test F1 Score", f1)
    mlflow.log_metric("Test Precision", precision)
    mlflow.log_metric("Test Recall", recall)
    mlflow.log_metric("Test ROC AUC", roc_auc)

    mlflow.sklearn.log_model(best_clf, artifact_path)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
    try:
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"Model registered with version: {result.version}")
    except Exception as e:
        print(f"Error registering model: {str(e)}")


# Main experiment function
def run_experiment(experiment_name, artifact_path, registered_model_name):
    start_time = time.time()

    # Load and preprocess data
    df = load_data()

    # Define columns for consistency after loading
    columns = ['date_ech', 'code_qual', 'lib_qual', 'coul_qual', 'date_dif', 'source', 'type_zone', 
               'code_zone', 'lib_zone', 'code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 
               'x_wgs84', 'y_wgs84', 'x_reg', 'y_reg', 'epsg_reg', 'ObjectId', 'x', 'y']
    df = pd.DataFrame(df, columns=columns)
    
    # Step 1: Convert 'date_ech' column to datetime and set it as the index
    df['date_ech'] = pd.to_datetime(df['date_ech'])
    df.set_index('date_ech', inplace=True)

    # Step 2: Check for duplicates in the timestamp index and handle them
    duplicates = df.index.duplicated().sum()
    print(f'Number of duplicate timestamps: {duplicates}')
    df = df.loc[~df.index.duplicated(keep='first')]  # Optionally drop duplicates

    # Resample the data to hourly frequency and forward fill missing values
    df = df.resample('H').ffill()

    # Pollutants list for ARIMA
    pollutants = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25']

    # Forecast pollutant levels
    predictions = forecast_pollutant_levels(df, pollutants)
    
    # Present forecasted results
    forecasted_results = pd.DataFrame(predictions)
    forecasted_results['timestamp'] = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=2, freq='H')
    print("\nForecasted pollutant levels for the next 2 hours:")
    print(forecasted_results)

    # Split the data for ML model training
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Setup MLflow experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Ensure no active runs are left open
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.sklearn.autolog()

    # Start a new MLflow run
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train RandomForest model with GridSearchCV
        clf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Log model and metrics
        log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)

    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")


# Entry point for the script
if __name__ == "__main__":
    mlflow.set_tracking_uri("https://veeramanicadas-mlflow-server.hf.space")
    experiment_name = "air_quality_tuning"
    mlflow.set_experiment(experiment_name)
    
    artifact_path = "air_quality_model"
    registered_model_name = "air_quality_random_forest"

    run_experiment(experiment_name, artifact_path, registered_model_name)
