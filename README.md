## Table of Contents
- [Problem Statement](#ProblemStatement)
- [Technologies Used](#TechnologiesUsed)
- [Workflow](#Workflow)
- [Usage](#usage)
- [Contributing](#contributing)
- [Pre-execution Steps](#Pre-executionSteps)

## Problem Statement:
We have a trained ML model that predicts the price of a house based on its square footage. Over time, the price of houses has increased, rendering the predictions of the model inaccurate. Thus, it needs to be retrained on new data to learn the updated relationship between a house's square footage and its price.

## Technologies Used:
**Scikit-Learn**: The model used for prediction is Linear Regression from the sklearn library. It was also used to split the data for training and testing. Additionally, it was used to evaluate the model and compare the predictions with the ground truth.

**MLFlow**: MLFlow is an open-source tool that assists in implementing a machine learning lifecycle. Some of its key features include experiment tracking, model selection and deployment, and projects. We will leverage MLFlow's model selection and deployment feature to fetch the latest model from the model registry.

**Streamlit**: Streamlit is an open-source tool to create a web app for our model. It provides an interface for us to draw predictions and check if model drift exists.

## Workflow:
![Project_Workflow](https://github.com/PranavKhedkar/ModelRetrainingUsingMLFlow/assets/99120112/a907a4a3-dedc-4f09-a4cb-fb61a9e5ac6c)

Steps:
1. We use the latest training data CSV file from our "Training Data" folder, which will be used to train the model.
2. We train our linear regression model with the latest data fetched from the "Training Data" folder.
3. This trained model is then registered in MLFlow's Model Registry.
4. To make predictions, we fetch this model from the Model Registry and provide the input from the user.
5. The model provides us with an output which is stored in predictions.csv along with the input given to it.
6. The user/data owner is requested to submit the actual price at which the house was sold. This value will be considered as the ground truth for evaluating our model. Along with this value, the input value for prediction will also be stored in groundtruth.csv.
7. Next, we compare the predictions made by the model with the ground truth using sklearn's accuracy_score.
8. A threshold of 75% is specified to initiate retraining. This threshold will be used to determine model quality and verify if data drift exists. If the accuracy is above or equal to 75%, the current version of the model will continue to be used for predictions.
9. If the accuracy falls below 75%, retraining is initiated using trainmodel.py.

If the accuracy falls below 75%, trainmodel.py is executed. This will first fetch the latest data from the Training Data folder (Step 1) and train the model on it (Step 2). This model is then stored as a new version in the MLFlow Model Registry (Step 3). As predictions are done using the latest model, henceforth this model will be fetched for making predictions (Step 4). This cycle repeats every time the "Check Data Drift" button is pressed.

## Pre-execution Steps
- Start MLFlow server
```
mlflow ui
```
- Run streamlit to view the interface
```
streamlit run frontend.py
```
