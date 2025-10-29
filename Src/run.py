import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def linear_regression(x, w, b):
    return x * w + b

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
torch.random
w = torch.randn(1)
b = torch.randn(1)



xr = torch.linspace(x_train.min(), x_train.max(), 100).unsqueeze(1)
y_hat = linear_regression(xr, w, b)
plt.scatter(x_train, y_train)
plt.plot(xr, y_hat, color='red')
plt.show()