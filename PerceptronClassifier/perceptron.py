import numpy as np

np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def perceptronPrediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = perceptronPrediction(X, W, b)
        if y_hat - y[i] == 1:
            W[0] -= learn_rate * X[i][0]
            W[1] -= learn_rate * X[i][1]
            b -= learn_rate
        elif y_hat - y[i] == -1:
            W[0] += learn_rate * X[i][0]
            W[1] += learn_rate * X[i][1]
            b += learn_rate
    return W, b

def perceptronTrain(X, y, num_epochs = 100, learn_rate = 0.01):
    x_max = max(X.T[0])
    x_min = min(X.T[0])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max

    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b)

    return -W[0]/W[1], -b/W[1] + 0.5*(x_min+x_max)

