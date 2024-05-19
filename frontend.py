import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import subprocess
import mlflow.pyfunc
from mlflow import MlflowClient
import os
import time
import math

def load_model():
    client = MlflowClient()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model_name = "LinearRegressor"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    model_info = client.get_latest_versions(model_name, stages=["None"])

    for m in model_info:
            print(f"name: {m.name}")
            print(f"Version: {m.version}")
            return model, m.name, m.version

def main():

    model, model_name, model_version = load_model()

    st.title("House Price Predictor")

    # Take input from the user
    Sqft = st.number_input("Enter the Sqft area of the house:", step=30)

    if st.button(label='Submit'):
        make_prediction(Sqft, model)
    
    ground_truth = st.number_input("Enter the actual price paid:", step=30)

    if st.button(label='Submit Actual Price'):
        store_groundtruth(Sqft, ground_truth)
    
    if st.button(label='Check for Drift', type="primary"):
        drift_detection()
    
    # Display Model Information
    style = """
    <style>
    .bottom-right {
    position: fixed;
    bottom: 10px;
    right: 10px;
    padding: 5px;
    # background-color: rgba(0, 0, 0, 0.5);
    color: white;
    }
    </style>
    """
    space = "&nbsp;&nbsp;&nbsp;&nbsp;"
    st.markdown(style, unsafe_allow_html=True)
    st.markdown(f"""<div class="bottom-right">Model Used: {space}{model_name}<br>
    Model Version: {space}1.{model_version}</div>""", unsafe_allow_html=True)

def make_prediction(Sqft, model):
    # Make prediction
    prediction = model.predict(pd.DataFrame({"Sqft": [Sqft]}))
    store_results(Sqft, prediction)
    st.write("Appropriate Price:    ₹",str(math.ceil(prediction[0])))

def store_results(Sqft, prediction):
    df_new = pd.DataFrame({"Input": [Sqft], "Prediction": math.ceil(prediction)})

    # Check if the file exists
    file_exists = os.path.isfile("predictions.csv")

    # Append the new predictions to the existing predictions or create a new file
    if not file_exists or os.stat("predictions.csv").st_size == 0:
        df_new.to_csv("predictions.csv", index=False)
    else:
        df_new.to_csv("predictions.csv", index=False, header=False, mode="a")

def store_groundtruth(Sqft, user_entered_value):
    df_new = pd.DataFrame({"Input": [Sqft], "groundtruth": user_entered_value})

    # Check if the file exists
    file_exists = os.path.isfile("groundtruth.csv")

    # Append the new predictions to the existing predictions or create a new file
    if not file_exists or os.stat("groundtruth.csv").st_size == 0:
        df_new.to_csv("groundtruth.csv", index=False)
    else:
        df_new.to_csv("groundtruth.csv", index=False, header=False, mode="a")

def drift_detection():
    accuracy_threshold = 0.75

    df_predictions = pd.read_csv("predictions.csv")
    df_groundtruth = pd.read_csv("groundtruth.csv")

    # Accuaracy
    accuracy = accuracy_score(df_predictions["Prediction"], df_groundtruth["groundtruth"])
    print("Accuracy:", accuracy)

    if accuracy < accuracy_threshold:
        st.write("Drift Detected")
        st.write("Model Accuracy:   ",str(accuracy*100),"%")
        print("________________RERUN_INTIATED________________")
        # Retrain the model
        subprocess.run(["D:\MLFlow\Retrain\.model_retrain\Scripts\python.exe", "trainmodel.py"])

        st.rerun()
    else:
        st.write("Drift Not Detected")
        st.write("Model Accuracy:   ",str(accuracy*100),"%")


if __name__ == "__main__":
    main()
