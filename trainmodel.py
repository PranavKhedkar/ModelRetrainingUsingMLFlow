"""

Author:  Pranav Khedkar
Date:    20 May 2024

"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.models import infer_signature
import glob
import os

# Load data
list_of_files = glob.glob('D:\MLFlow\Housing Price - Retrain\Training Data\*.csv') 
latest_file = max(list_of_files, key=os.path.getctime)  # Get latest file
df = pd.read_csv(latest_file)

# Split features and labels
X = df[['Sqft']]
y = df['Price']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Specify Parameters
params = {
    "fit_intercept": True,
}

# Model training
model = LinearRegression(**params)
model.fit(X_train, y_train)

# Model prediction
pred = model.predict(X_test)

# Model performance
mse = mean_squared_error(y_true=y_test, y_pred=pred)
rmse = np.sqrt(mse)

# Log model
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Model Retraining - House Price")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the performance metric
    mlflow.log_metric("rmse", rmse)

    # Set a tag
    mlflow.set_tag("Training", "LinearRegessor")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="House_price",
        input_example=X_train.head(),
        signature=signature,
        registered_model_name="LinearRegressor",
    )

