# import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_name = "RandomForest"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

data = np.array([[120]])
results = model.predict(data)

print("Prediction:  ",results)

# Create a DataFrame with input values and predictions
df_new = pd.DataFrame({"Input": data.flatten(), "Prediction": results})

# Check if the file exists
file_exists = os.path.isfile("predictions.csv")

# Append the new predictions to the existing predictions or create a new file
with open("predictions.csv", "a") as f:
    if not file_exists or os.stat("predictions.csv").st_size == 0:
        df_new.to_csv(f, index=False)
    else:
        df_new.to_csv(f, index=False, header=False)
