import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def linear_regression(x, w, b):
    return x * w + b

# Loss Function
def mse(y_hat, y):
    return torch.mean((y_hat - y) ** 2)

def gradient_descent(x, y, y_hat, w, b, lr=0.01):
    N = len(y)
    dw = (-2/N) * torch.sum(x * (y - y_hat))
    db = (-2/N) * torch.sum(y - y_hat)

    w -= lr * dw
    b -= lr * db

    return w, b

salery = pd.read_csv("./Data/Salary Data.csv")

X = salery['Experience Years'].values
Y = salery['Salary'].values

x_train, x_test , y_train, y_test= train_test_split(X, Y, test_size=0.3)

xScaler = StandardScaler()
yScaler = StandardScaler()

x_train = xScaler.fit_transform(x_train.reshape(-1, 1))
y_train = yScaler.fit_transform(y_train.reshape(-1, 1))
x_test = xScaler.transform(x_test.reshape(-1, 1))
y_test = yScaler.transform(y_test.reshape(-1, 1))

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


w = torch.randn(1)
b = torch.randn(1)

for i in range(100):
    y_hat = linear_regression(x_train, w, b)
    loss = mse(y_hat, y_train)
    w, b = gradient_descent(x_train, y_train, y_hat, w, b, lr=0.1)

y_pred = linear_regression(x_test, w, b)
test_loss = mse(y_pred, y_test)
print(f"Test MSE Loss: {test_loss.item()}")