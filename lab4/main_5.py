import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
data = fetch_california_housing()
scaler = MinMaxScaler()
x_data = np.array(data['data'])
scaler.fit(x_data)
x_norm = scaler.transform(x_data)
y = np.array(data['target'])

def func(train_s):
    X_train, X_test, y_train, y_test = train_test_split(x_norm, y, train_size=train_s, test_size=1-train_s)
    network1 = MLPRegressor(solver='adam',hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter = 500, tol = 0.001, activation = 'tanh')
    network1.fit(X_train, y_train)
    y_pred1 = network1.predict(X_test)
    mse1 = mean_squared_error(y_test, y_pred1)
    score1 = network1.score(X_test, y_test)
    print('Layer (100, 100, 100, 100, 100), activation tanh, network score: ' + str(score1) + ', mean squared error: ' + str(mse1) + ' ITER 500' + ', train/test ' + str(train_s) + '/' + str(1-train_s))

func(0.20)
func(0.35)
func(0.50)
func(0.65)
func(0.80)