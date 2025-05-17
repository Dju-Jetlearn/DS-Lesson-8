import matplotlib.pyplot as matpat
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/diego/JetLearn/Data Science/Lesson 8/Titanic.csv")

print(data.head(10))
print(data.info())

# Data Preprocessing

data["Sex"] = data["Sex"].replace({"male": 0, "female": 1})

print(data.head())

# Data Analysis

# Input & Output

x_data_1 = data.drop("Survived", axis = 1)
x_data_2 = data.drop("Name", axis = 1)


x = x_data_1 + x_data_2

y = data["Survived"]

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

print("Accuracy Percentage =", accuracy_percent, "%")