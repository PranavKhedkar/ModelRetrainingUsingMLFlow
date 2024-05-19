# Predict using registered model

# import mlflow.sklearn
# import numpy as np

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# # Load the model by specifying the URI of the registered model
# model_uri = "runs:/e817e7a3c9b84316ad86a354cdf2eef4/movie_data"
# model = mlflow.sklearn.load_model(model_uri)

# # Example input for prediction
# X_new = np.array([[-1], [300]])

# # Make predictions
# predictions = model.predict(X_new)

# print("Predictions:", predictions)


#-------------------------------------------------------------------------------

# from mlflow import MlflowClient

# client = MlflowClient()
# model_name = "RandomForest"
# latest_versions = client.search_model_versions(filter_string=f"name='{model_name}'")
# print(latest_versions)

# model_info = client.get_latest_versions(model_name, stages=["None"])

# for m in model_info:
#         print(f"name: {m.name}")
#         print(f"Version: {m.version}")
# print(client.get_latest_versions(model_name, stages=["None"]))

import glob
import os
import pandas as pd

list_of_files = glob.glob('D:\MLFlow\Retrain\Training Data\*.csv') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
df = pd.read_csv(latest_file)
print(latest_file)