# -*- coding: utf-8 -*-
"""frontend.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SVuACDbXfRHEjMnxvWDHenSyntVGB3UZ
"""

# Install necessary libraries
!pip install -q ipywidgets

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset from CSV file
df = pd.read_csv('heart.csv')  # Update 'heart.csv' with your dataset file name

# Display the first few rows of the dataset
display(df.head())

# Extract features (X) and target (y) from the dataset
X = df.drop('target', axis=1)  # Assuming 'target' is the column name for the target variable
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define machine learning models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

# Function to train and evaluate a selected model
def train_and_evaluate_model(model_name):
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Display accuracy and confusion matrix
    with output:
        clear_output()  # Clear previous output
        print(f"Accuracy of {model_name}: {accuracy:.2f}\n")
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix ({model_name})")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], labels=["No Disease", "Disease"])
        plt.yticks([0, 1], labels=["No Disease", "Disease"])
        plt.show()

# Dropdown widget for model selection
model_dropdown = widgets.Dropdown(
    options=list(models.keys()),
    description='Select Model:'
)

# Button widget to trigger model training and evaluation
train_button = widgets.Button(
    description='Train Model',
    button_style='info'
)

# Output widget to display results
output = widgets.Output()

# Define function to handle button click event
def on_button_click(b):
    model_name = model_dropdown.value
    train_and_evaluate_model(model_name)

# Attach button click event
train_button.on_click(on_button_click)

# Display widgets
display(model_dropdown)
display(train_button)
display(output)

# Input widgets for feature values
age_input = widgets.IntSlider(min=20, max=80, step=1, description='Age:')
sex_input = widgets.Dropdown(options=[('Male', 1), ('Female', 0)], description='Sex:')
cp_input = widgets.Dropdown(options=[('Typical Angina', 0), ('Atypical Angina', 1), ('Non-anginal Pain', 2), ('Asymptomatic', 3)], description='Chest Pain Type:')
trestbps_input = widgets.IntSlider(min=80, max=200, step=1, description='Resting Blood Pressure:')
chol_input = widgets.IntSlider(min=100, max=600, step=1, description='Cholesterol:')
fbs_input = widgets.Dropdown(options=[('Normal', 0), ('High', 1)], description='Fasting Blood Sugar:')
restecg_input = widgets.Dropdown(options=[('Normal', 0), ('ST-T Wave Abnormality', 1), ('Probable Left Ventricular Hypertrophy', 2)], description='Resting ECG:')
thalach_input = widgets.IntSlider(min=60, max=220, step=1, description='Max Heart Rate:')
exang_input = widgets.Dropdown(options=[('No', 0), ('Yes', 1)], description='Exercise Induced Angina:')
oldpeak_input = widgets.FloatSlider(min=0.0, max=6.0, step=0.1, description='ST Depression:')
slope_input = widgets.Dropdown(options=[('Upsloping', 0), ('Flat', 1), ('Downsloping', 2)], description='Slope:')
ca_input = widgets.Dropdown(options=[('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4)], description='Number of Major Vessels:')
thal_input = widgets.Dropdown(options=[('Normal', 0), ('Fixed Defect', 1), ('Reversible Defect', 2)], description='Thalassemia:')

# Button widget to trigger prediction
predict_button = widgets.Button(description='Submit', button_style='success')
prediction_output = widgets.Output()

# Define function to handle prediction
def on_predict_button_click(b):
    with prediction_output:
        clear_output()
        # Gather input values
        instance = [
            age_input.value, sex_input.value, cp_input.value, trestbps_input.value,
            chol_input.value, fbs_input.value, restecg_input.value, thalach_input.value,
            exang_input.value, oldpeak_input.value, slope_input.value, ca_input.value,
            thal_input.value
        ]
        # Reshape input for prediction
        instance = np.array(instance).reshape(1, -1)
        # Select model (e.g., Random Forest) for prediction
        model = models['Random Forest']  # Use Random Forest model for prediction
        model.fit(X_train, y_train)
        prediction = model.predict(instance)
        if prediction[0] == 1:
            print("The person is likely to have heart disease.")
        else:
            print("The person is not likely to have heart disease.")

# Attach button click event for prediction
predict_button.on_click(on_predict_button_click)

# Display input widgets and prediction button
display(age_input, sex_input, cp_input, trestbps_input, chol_input, fbs_input, restecg_input,
        thalach_input, exang_input, oldpeak_input, slope_input, ca_input, thal_input)
display(predict_button)
display(prediction_output)