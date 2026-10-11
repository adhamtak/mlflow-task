import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_uri = "models:/California_Housing_Predictor@staging"
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path="./model")
print("Model downloaded successfully!")