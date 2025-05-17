import matplotlib.pyplot as matpat
import pandas as pd

data = pd.read_csv("C:/Users/diego/JetLearn/Data Science/Lesson 8/iris.csv")

data["species"] = data["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2})

w = data["sepal_width"]
x = data["sepal_length"]
y = data["petal_width"]
z = data["petal_length"]
a = data["species"]

matpat.scatter(w, a, label = " Iris scatter plot", color = "red", marker = '^', s = 100)
matpat.xlabel("Sepal Width")
matpat.ylabel("Species")
matpat.title("Iris scatter plot")
matpat.show()

matpat.scatter(x, a, label = " Iris scatter plot", color = "blue", marker = 'o', s = 100)
matpat.xlabel("Sepal length")
matpat.ylabel("Species")
matpat.title("Iris scatter plot")
matpat.show()

matpat.scatter(y, a, label = " Iris scatter plot", color = "green", marker = '^', s = 100)
matpat.xlabel("Petal Width")
matpat.ylabel("Species")
matpat.title("Iris scatter plot")
matpat.show()

matpat.scatter(z, a, label = " Iris scatter plot", color = "purple", marker = 'o', s = 100)
matpat.xlabel("Petal length")
matpat.ylabel("Species")
matpat.title("Iris scatter plot")
matpat.show()