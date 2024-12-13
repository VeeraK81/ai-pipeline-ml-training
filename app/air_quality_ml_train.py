# import pandas as pd
# import numpy as np
# import mlflow
# import time
# import boto3
# from io import StringIO
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller
# import os
# from mlflow.tracking import MlflowClient


# # Sample data loading function from S3 (this can be replaced with file reading)
# def load_data():
#     bucket_name = os.getenv('BUCKET_NAME')
#     file_key = os.getenv('FILE_KEY')

#     # Create an S3 client
#     s3_client = boto3.client(
#         's3',
#         aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#         aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
#     )

#     # Read the CSV file from S3
#     response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
#     csv_content = response['Body'].read().decode('utf-8')
    
#     # Load into a pandas DataFrame
#     return pd.read_csv(StringIO(csv_content))


# # Preprocess data function
# def preprocess_data(df):
#     features = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 'x_wgs84', 'y_wgs84']
#     target = 'code_qual'
    
#     # Extract features and target
#     X = df[features]
#     y = df[target]

#     # Handling NaN and infinite values in features and target
#     X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
#     X = X.fillna(X.mean())  # Impute NaN values with the mean of each column
    
#     # If the target variable has NaN values, you can either impute or drop them (depends on your choice)
#     y = y.dropna()  # Drop rows with NaN in the target variable

#     # Align the features and target after dropping NaN rows
#     X = X.loc[y.index]  # Align X with the cleaned y

#     # Split the data
#     return train_test_split(X, y, test_size=0.2, random_state=42)


# def forecast_pollutant_levels(df, pollutants):
#     predictions = {}
#     for pollutant in pollutants:
#         # Handle NaN values in the pollutant series
#         pollutant_data = df[pollutant].dropna()  # Drop NaN values from the pollutant data
        
#         # Handle infinite values by replacing with large number
#         pollutant_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#         pollutant_data = pollutant_data.fillna(pollutant_data.mean())  # Impute missing values with the mean

#         # Now fit the ARIMA model
#         if pollutant_data.nunique() == 1:
#             print(f"Handling constant values for {pollutant}. Forecasted values will be the observed value.")
#             predictions[pollutant] = np.full(2, pollutant_data.iloc[-1])
#         else:
#             print(f"Fitting ARIMA model for {pollutant}...")
#             try:
#                 model = ARIMA(pollutant_data, order=(1, 1, 1))
#                 model_fit = model.fit()
#                 forecast = model_fit.forecast(steps=2)
#                 predictions[pollutant] = forecast
#                 print(f"Forecasted {pollutant} levels for the next 2 hours: {forecast}")
#             except Exception as e:
#                 print(f"Error fitting ARIMA model for {pollutant}: {e}")
#                 predictions[pollutant] = None

#     return predictions


# # Function to log the model and metrics to MLflow
# def log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
#     best_clf = grid_search.best_estimator_
#     y_pred = best_clf.predict(X_test)

#     f1 = f1_score(y_test, y_pred, average='weighted')
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test), multi_class='ovr', average='weighted')

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
#     print("Accuracy Score:", accuracy_score(y_test, y_pred))

#     mlflow.log_metric("Train Accuracy", accuracy_score(y_train, best_clf.predict(X_train)))
#     mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
#     mlflow.log_metric("Test F1 Score", f1)
#     mlflow.log_metric("Test Precision", precision)
#     mlflow.log_metric("Test Recall", recall)
#     mlflow.log_metric("Test ROC AUC", roc_auc)

#     mlflow.sklearn.log_model(best_clf, artifact_path)

#     model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
#     run_id = mlflow.active_run().info.run_id
#     try:
#         # result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
#         client = MlflowClient()
#         # Register the model
#         try:
#             client.create_registered_model(registered_model_name)
#             client.create_model_version(name=registered_model_name, source=model_uri, run_id=run_id)
#             print(f"Model registered with version: {run_id}")
#         except:
#             client.create_model_version(name=registered_model_name, source=model_uri, run_id=run_id)
#             print(f"Model registered with version: {run_id}")
#     except Exception as e:
#         print(f"Error registering model: {str(e)}")


# # Main experiment function
# def run_experiment(experiment_name, artifact_path, registered_model_name):
#     start_time = time.time()

#     # Load and preprocess data
#     df = load_data()

#     # Define columns for consistency after loading
#     columns = ['date_ech', 'code_qual', 'lib_qual', 'coul_qual', 'date_dif', 'source', 'type_zone', 
#                'code_zone', 'lib_zone', 'code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25', 
#                'x_wgs84', 'y_wgs84', 'x_reg', 'y_reg', 'epsg_reg', 'ObjectId', 'x', 'y']
#     df = pd.DataFrame(df, columns=columns)
    
#     # Step 1: Convert 'date_ech' column to datetime and set it as the index
#     df['date_ech'] = pd.to_datetime(df['date_ech'])
#     df.set_index('date_ech', inplace=True)

#     # Step 2: Check for duplicates in the timestamp index and handle them
#     duplicates = df.index.duplicated().sum()
#     print(f'Number of duplicate timestamps: {duplicates}')
#     df = df.loc[~df.index.duplicated(keep='first')]  # Optionally drop duplicates

#     # Resample the data to hourly frequency and forward fill missing values
#     df = df.resample('H').ffill()

#     # Pollutants list for ARIMA
#     pollutants = ['code_no2', 'code_so2', 'code_o3', 'code_pm10', 'code_pm25']

#     # Forecast pollutant levels
#     predictions = forecast_pollutant_levels(df, pollutants)
    
#     # Present forecasted results
#     forecasted_results = pd.DataFrame(predictions)
#     forecasted_results['timestamp'] = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=2, freq='H')
#     print("\nForecasted pollutant levels for the next 2 hours:")
#     print(forecasted_results)

#     # Split the data for ML model training
#     X_train, X_test, y_train, y_test = preprocess_data(df)

#     # Setup MLflow experiment
#     mlflow.set_experiment(experiment_name)
#     experiment = mlflow.get_experiment_by_name(experiment_name)

#     # Ensure no active runs are left open
#     if mlflow.active_run():
#         mlflow.end_run()

#     mlflow.sklearn.autolog()

#     # Start a new MLflow run
#     with mlflow.start_run(experiment_id=experiment.experiment_id):
#         # Train RandomForest model with GridSearchCV
#         clf = RandomForestClassifier(random_state=42)

#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [10, 20, 30, None],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': ['sqrt', 'log2']
#         }

#         grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
#         grid_search.fit(X_train, y_train)

#         # Log model and metrics
#         log_model_and_metrics(grid_search, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)

#     print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")


# # Entry point for the script
# if __name__ == "__main__":
#     mlflow.set_tracking_uri("https://veeramanicadas-mlflow-server.hf.space")
#     experiment_name = "air_quality_tuning"
#     mlflow.set_experiment(experiment_name)
#     artifact_path = "models/air_quality_model"
#     registered_model_name = "air_quality_random_forest"

#     run_experiment(experiment_name, artifact_path, registered_model_name)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import pickle
import boto3
from io import StringIO


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
    
    data = pd.read_csv(StringIO(csv_content))
    data['date_debut'] = pd.to_datetime(data['date_debut'])
    
    return data

def preprocessing(data):
    """
    Preprocess the dataset by feature engineering and encoding.
    """
    data = data.sort_values(by='date_debut')
    data = data[['date_debut', 'nom_station', 'typologie', 'influence', 'valeur', 'x_wgs84', 'y_wgs84']].dropna()
    data['hour'] = data['date_debut'].dt.hour
    data['day_of_week'] = data['date_debut'].dt.dayofweek
    data['month'] = data['date_debut'].dt.month

    data = pd.get_dummies(data, columns=['nom_station', 'typologie', 'influence'], drop_first=True)
    
    return data

def create_pipeline():
    """
    Define the model pipeline and hyperparameter grid for tuning.
    """
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    return grid_search

# def save_model_and_features(model, feature_columns, model_path="models/no2_model.pkl", features_path="models/feature_columns.pkl"):
#     # Save the trained model
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)
#     print(f"Model saved to {model_path}")

#     # Save the feature names
#     with open(features_path, 'wb') as f:
#         pickle.dump(feature_columns, f)
#     print(f"Feature columns saved to {features_path}")
    
    

def train_model(data, experiment):
    """
    Train the model with hyperparameter tuning and log the results with MLflow.
    """
    # Select features for training
    features = [col for col in data.columns if col not in ['valeur', 'date_debut']]
    X = data[features]
    y = data['valeur']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Best Parameters:", pipeline.best_params_)
    print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
    
    # Log metrics and parameters to MLflow
    expDetail = mlflow.get_experiment_by_name(experiment)
    with mlflow.start_run(experiment_id=expDetail.experiment_id):
        mlflow.log_param("best_params", pipeline.best_params_)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipeline.best_estimator_, "model")
        print("Model and metrics logged to MLflow.")



# Main function
if __name__ == "__main__":
    file_path = "air_quality_data_new.csv"
    mlflow.set_tracking_uri("https://veeramanicadas-mlflow-server.hf.space")
    # Log metrics and model with MLflow
    experiment_name = "air_quality_tuning"
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    data = load_data()
    data = preprocessing(data)
    train_model(data, experiment_name)