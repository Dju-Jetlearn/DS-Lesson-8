import matplotlib.pyplot as matpat
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/diego/JetLearn/Data Science/Lesson 8/iris.csv")

print(data.head(10))
print(data.info())

data["species"] = data["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2})

print(data.head())