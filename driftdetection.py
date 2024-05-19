import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import subprocess

accuracy_threshold = 0.75

df_predictions = pd.read_csv("predictions.csv")
df_groundtruth = pd.read_csv("groundtruth.csv")

# Accuaracy
accuracy = accuracy_score(df_predictions["Prediction"], df_groundtruth["groundtruth"])
print("Accuracy:", accuracy)

if accuracy < accuracy_threshold:
    subprocess.run(["D:\MLFlow\Retrain\.model_retrain\Scripts\python.exe", "trainmodel.py"])
else:
    print("Model Drift Not Detected")