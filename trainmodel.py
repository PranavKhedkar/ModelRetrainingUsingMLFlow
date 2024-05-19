import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature
from requests.exceptions import ConnectionError
import glob
import os

# Load data

list_of_files = glob.glob('D:\MLFlow\Retrain\Training Data\*.csv') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print("latest_file: ",latest_file)
df = pd.read_csv(latest_file)

# Split features and labels
X = df[['Duration']]
y = df['Long']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

params = {
    "max_depth": 2,
    "random_state": 0
}

# Model training
clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)

# Model prediction
pred = clf.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_true=y_test,y_pred=pred)

# Log model
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Model Retraining")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training", "Random Forest Classifier")

    # Infer the model signature
    signature = infer_signature(X_train, clf.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="movie_data",
        input_example=X_train.head(),
        signature=signature,
        registered_model_name="RandomForest",
    )

print(model_info)

