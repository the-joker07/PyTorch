import torch
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
salery = pd.read_csv("./Data/Salary Data.csv")

X = salery['Experience Years'].values
Y = salery['Salary'].values

xScaler = StandardScaler()
yScaler = StandardScaler() 

x_train, x_test , y_train, y_test= train_test_split(X, Y, test_size=0.2)