import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import subprocess
import mlflow.pyfunc
from mlflow import MlflowClient
import os
import time

def load_model():
    client = MlflowClient()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model_name = "RandomForest"
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

    st.title("MlFlow for Retraining")

    # Take input from the user
    duration = st.number_input("Enter the duration of the movie:", step=30)

    if st.button(label='Submit'):
        make_prediction(duration, model)
    
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

def make_prediction(duration, model):
    # Make prediction
    prediction = model.predict(pd.DataFrame({"Duration": [duration]}))
    store_results(duration, prediction)

    # Show prediction
    if prediction[0] == 1:
        st.write("This is a long movie.")
    else:
        st.write("This is a short movie.")

def store_results(duration, prediction):
    df_new = pd.DataFrame({"Input": [duration], "Prediction": prediction})

    # Check if the file exists
    file_exists = os.path.isfile("predictions.csv")

    # Append the new predictions to the existing predictions or create a new file
    if not file_exists or os.stat("predictions.csv").st_size == 0:
        df_new.to_csv("predictions.csv", index=False)
    else:
        df_new.to_csv("predictions.csv", index=False, header=False, mode="a")

def drift_detection():
    accuracy_threshold = 0.75

    df_predictions = pd.read_csv("predictions.csv")
    df_groundtruth = pd.read_csv("groundtruth.csv")

    # Accuaracy
    accuracy = accuracy_score(df_predictions["Prediction"], df_groundtruth["groundtruth"])
    print("Accuracy:", accuracy)

    if accuracy < accuracy_threshold:
        st.write("Drift Detected")
        print("______RERUN_INTIATED__________")
        # Retrain the model
        subprocess.run(["D:\MLFlow\Retrain\.model_retrain\Scripts\python.exe", "trainmodel.py"])

        st.rerun()
    else:
        st.write("Drift Not Detected")


if __name__ == "__main__":
    main()
