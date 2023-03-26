import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

data = fetch_california_housing()
scaler = MinMaxScaler()
x_data = np.array(data['data'])
scaler.fit(x_data)
x_norm = scaler.transform(x_data)
y = np.array(data['target'])

X_train, X_test, y_train, y_test = train_test_split(x_norm, y)

network1 = MLPRegressor(solver='adam',hidden_layer_sizes=(20, 20), max_iter = 5000, tol = 0.001, activation = 'relu')
network1.fit(X_train, y_train)
y_pred1 = network1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
score1 = network1.score(X_test, y_test)
print('Layer (20, 20), activation relu, network score: ' + str(score1) + ', mean squared error: ' + str(mse1))

network2 = MLPRegressor(solver='adam',hidden_layer_sizes=(20, 20), max_iter = 5000, tol = 0.001, activation = 'tanh')
network2.fit(X_train, y_train)
y_pred2 = network2.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)
score2 = network2.score(X_test, y_test)
print('Layer (20, 20), activation tanh, network score: ' + str(score2) + ', mean squared error: ' + str(mse2))
    
network3 = MLPRegressor(solver='adam',hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter = 5000, tol = 0.001, activation = 'relu')
network3.fit(X_train, y_train)
y_pred3 = network3.predict(X_test)
mse3 = mean_squared_error(y_test, y_pred3)
score3 = network3.score(X_test, y_test)
print('Layer (100, 100, 100, 100, 100), activation relu, network score: ' + str(score3) + ', mean squared error: ' + str(mse3))

   
network4 = MLPRegressor(solver='adam',hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter = 5000, tol = 0.001, activation = 'tanh')
network4.fit(X_train, y_train)
y_pred4 = network4.predict(X_test)
mse4 = mean_squared_error(y_test, y_pred4)
score4 = network4.score(X_test, y_test)
print('Layer (100, 100, 100, 100, 100), activation tanh, network score: ' + str(score4) + ', mean squared error: ' + str(mse4))

