"""

Author:  Pranav Khedkar
Date:    20 May 2024

"""

# Import required libraries
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import subprocess
import mlflow.pyfunc
from mlflow import MlflowClient
import os
import math

# Load the lastest version of the model from MLFlow
def load_model():
    client = MlflowClient()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Fetch if model already exists
    try:
        model_name = "LinearRegressor"
        model_version = "latest"

        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)

        model_info = client.get_latest_versions(model_name, stages=["None"])

        for m in model_info:
                print(f"name: {m.name}")
                print(f"Version: {m.version}")
                return model, m.name, m.version
    except:                                     # Create experiment and train first model if not already exists
        subprocess.run(["hprice\Scripts\python.exe", "trainmodel.py"])
        model_name = "LinearRegressor"
        model_version = "latest"

        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)

        model_info = client.get_latest_versions(model_name, stages=["None"])

        for m in model_info:
                print(f"name: {m.name}")
                print(f"Version: {m.version}")
                return model, m.name, m.version

# Make prediction
def make_prediction(Sqft, model):
    prediction = model.predict(pd.DataFrame({"Sqft": [Sqft]}))  # use the model and predict
    store_results(Sqft, prediction)     # store predictions by calling the store_results function
    st.write("Appropriate Price:    ₹",str(math.ceil(prediction[0])))   # display output

# Store predictions mode by the model
def store_results(Sqft, prediction):
    df_new = pd.DataFrame({"Input": [Sqft], "Prediction": math.ceil(prediction)})

    # Check if the file exists
    file_exists = os.path.isfile("predictions.csv")

    # Append the new predictions to the existing predictions or create a new file
    if not file_exists or os.stat("predictions.csv").st_size == 0:
        df_new.to_csv("predictions.csv", index=False)
        st.toast('Prediction Stored!')
    else:
        df_new.to_csv("predictions.csv", index=False, header=False, mode="a")
        st.toast('Prediction Stored!')

def store_groundtruth(Sqft, user_entered_value):
    df_new = pd.DataFrame({"Input": [Sqft], "groundtruth": user_entered_value})

    # Check if the file exists
    file_exists = os.path.isfile("groundtruth.csv")

    # Append the new predictions to the existing predictions or create a new file
    if not file_exists or os.stat("groundtruth.csv").st_size == 0:
        df_new.to_csv("groundtruth.csv", index=False)
        st.toast('Ground Truth Stored!')
    else:
        df_new.to_csv("groundtruth.csv", index=False, header=False, mode="a")
        st.toast('Ground Truth Stored!')

def drift_detection():
    
    # Set a accuracy threshold
    accuracy_threshold = 0.75

    # Fetch predictions and groundtruth
    df_predictions = pd.read_csv("predictions.csv")
    df_groundtruth = pd.read_csv("groundtruth.csv")

    # Accuaracy
    accuracy = accuracy_score(df_predictions["Prediction"], df_groundtruth["groundtruth"])
    print("Accuracy:", accuracy)

    try:
        accuracy = accuracy_score(df_predictions["Prediction"], df_groundtruth["groundtruth"])
        print("Accuracy:", accuracy)
        if accuracy < accuracy_threshold:
            st.write("Drift Detected")
            st.write("Model Accuracy:   ",str(accuracy*100),"%")
            print("________________RERUN_INTIATED________________")
            # Retrain the model
            subprocess.run(["hprice\Scripts\python.exe", "trainmodel.py"])
            
            # Refresh the page to update the model version displayed
            st.rerun()
        else:
            st.write("Drift Not Detected")
            st.write("Model Accuracy:   ",str(accuracy*100),"%")
    except ValueError:             # If number of entries in predictions.csv and groundtruth.csv do not match, raise exception
        st.toast('Number of Predictions and Ground Truth values do not match!')

def clear_files():
    try:
        if os.path.isfile("predictions.csv"):
            open("predictions.csv", "w").close()
        if os.path.isfile("groundtruth.csv"):
            open("groundtruth.csv", "w").close()
        st.toast("Entries deleted successfully!")
    except Exception as e:
        st.write(f"An error occurred: {e}")

# Define the streamlit frontend
def main():

    model, model_name, model_version = load_model()

    st.title("House Price Predictor")   # title

    # Take input from the user
    Sqft = st.number_input("Enter the Sqft area of the house:", step=30)

    # When submitted, call the make_prediction function
    if st.button(label='Submit'):
        make_prediction(Sqft, model)    
    
    # Take input from the user
    ground_truth = st.number_input("Enter the actual price paid:", step=30)

    # When submitted using "Submit Actual Price", call the store_groundtruth function and store it in groundtruth.csv
    if st.button(label='Submit Actual Price'):
        store_groundtruth(Sqft, ground_truth)   
    
    # When clicked, check for drift by calling drift_detection function
    if st.button(label='Check for Drift', type="primary"):
        drift_detection()

    with st.sidebar:
        st.write('Erase Predictions and Ground Truth:')
        if st.button(label="Erase"):
            clear_files()   
    
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

if __name__ == "__main__":
    main()
