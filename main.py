import matplotlib.pyplot as matpat
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/diego/JetLearn/Data Science/Lesson 8/iris.csv")

print(data.head(10))
print(data.info())

# Data Preprocessing

data["species"] = data["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2})

print(data.head())

# Data Analysis

# Input & Output

x = data.drop("species", axis = 1)

y = data["species"]

# Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

#Gives dimensions

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Algorithm

from sklearn.tree import DecisionTreeClassifier

# Initializing the algorithm

model = DecisionTreeClassifier(max_depth = 3, random_state = 1)

# Fitting Training data to the model

model.fit(x_train, y_train)

prediction = model.predict(x_test)

# Finding accuracy of Machine Learning Algorithm

from sklearn import metrics

accuracy = metrics.accuracy_score(prediction, y_test)

print("Accuracy =", accuracy)

accuracy_percent = accuracy*100

print("Accuracy Percentage =", accuracy_percent + "%")
