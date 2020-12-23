from perceptron import perceptronTrain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
X = data.T[0:2].T
y = data.T[2:3].T
X = np.array(X)
y = np.array(y)
k, b = perceptronTrain(X, y)
print(k, b)
for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(X[i][0], X[i][1], marker='o', color='red')
    else:
        plt.scatter(X[i][0], X[i][1], marker='s', color='blue')

x_lin = np.linspace(-1, 2, 20)
y_lin = x_lin * k + b
plt.plot(x_lin, y_lin, color='black')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()