import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading the csv data to a pandas Dataframe
heart_data=pd.read_csv("/content/hearts.csv")

# print first five rows of the dataset
heart_data.head()

# print last five rows of the dataset
heart_data.tail()

#number of rows and columns of dataset
heart_data.shape

# checking for missing values
heart_data.isnull().sum()

# statistical measure about the data
heart_data.describe()

# checking the distribution of target variables
heart_data["target"].value_counts()

X=heart_data.drop(columns="target",axis=1)
Y=heart_data["target"]

print(X)

print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

import pandas as pd

missing_values = X_train.isnull().sum()

if missing_values.any():
    print("There are missing values in the X_train data.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
hdata = pd.read_csv("/content/hearts.csv")

# Handle missing values for oldpeak and slope by filling them with the mean of the column
hdata['oldpeak'].fillna((hdata['oldpeak'].mean()), inplace=True)
hdata['slope'].fillna((hdata['slope'].mean()), inplace=True)
# Define the features and target variables
X = hdata.drop(['target'], axis=1)
y = hdata['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the logistic regression model
lr = LogisticRegression(max_iter=3000)
lr.fit(X_train, y_train)
#accuracy on training data
X_train_prediction=lr.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
# Predict on the test set
prediction = lr.predict(X_test)
# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, prediction)
confusion = confusion_matrix(y_test, prediction)
print("Training accuracy of the lr model=",training_data_accuracy)
print("Accuracy of the lr model is =", accuracy)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# Load the dataset
hdata = pd.read_csv("/content/hearts.csv")

# Handle missing values for oldpeak and slope by filling them with the mean of the column
hdata['oldpeak'].fillna((hdata['oldpeak'].mean()), inplace=True)
hdata['slope'].fillna((hdata['slope'].mean()), inplace=True)

# Define the features and target variables
X = hdata.drop(['target'], axis=1)
y = hdata['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
#accuracy on training data
X_train_prediction=svm.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
print("The Training accuracy for svm=",training_data_accuracy)
# Predict on the test set
y_pred = svm.predict(X_test)
# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Accuracy of the SVM model is =", accuracy)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the heart disease dataset
hdata = pd.read_csv('/content/hearts.csv')

# Handle missing values for oldpeak and slope by filling them with the mean of the column
hdata['oldpeak'].fillna((hdata['oldpeak'].mean()), inplace=True)
hdata['slope'].fillna((hdata['slope'].mean()), inplace=True)

# Split the data into features and target
X = hdata.drop('target', axis=1)
y = hdata['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf= RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=8, min_samples_split=5)
rf.fit(X_train, y_train)
#accuracy on training data
X_train_prediction=rf.predict(X_train)
print("The Training accuracy for random forest=",training_data_accuracy)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
# Make predictions on the test set
y_pred = rf.predict(X_test)
# Calculate the accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Accuracy of the Random Forest model is =", accuracy)

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from Kaggle
hdata = pd.read_csv("/content/hearts.csv")

# Display the first few rows of the dataset
print(hdata.head())

# Split the data into features (X) and target (y)
X = hdata.drop('target', axis=1)
y = hdata['target']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a decision tree classifier
dt = DecisionTreeClassifier()
# Train the classifier on the training data
dt.fit(X_train, y_train)
# Make predictions on the test data
y_pred = dt.predict(X_test)
#accuracy on training data
X_train_prediction=dt.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
print("The Training accuracy for Decision Tree=",training_data_accuracy)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of DT model: {accuracy}")

