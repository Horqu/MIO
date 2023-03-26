import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

with open('Advertising.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    var1 = []
    var2 = []
    var3 = []
    y1 = []
    for row in reader:
        var1.append(row[1])
        var2.append(row[2])
        var3.append(row[3])
        y1.append(row[4])

    x = np.array([[x1, x2, x3] for x1, x2, x3 in zip(var1, var2, var3)], dtype=np.float32)
    y = np.array(y1, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(x, y)

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
    
    network3 = MLPRegressor(solver='adam',hidden_layer_sizes=(1000, 1000, 1000, 1000, 1000), max_iter = 5000, tol = 0.001, activation = 'relu')
    network3.fit(X_train, y_train)
    y_pred3 = network3.predict(X_test)
    mse3 = mean_squared_error(y_test, y_pred3)
    score3 = network3.score(X_test, y_test)
    print('Layer (1000, 1000, 1000, 1000, 1000), activation relu, network score: ' + str(score3) + ', mean squared error: ' + str(mse3))

   
    network4 = MLPRegressor(solver='adam',hidden_layer_sizes=(1000, 1000, 1000, 1000, 1000), max_iter = 5000, tol = 0.001, activation = 'tanh')
    network4.fit(X_train, y_train)
    y_pred4 = network4.predict(X_test)
    mse4 = mean_squared_error(y_test, y_pred4)
    score4 = network4.score(X_test, y_test)
    print('Layer (1000, 1000, 1000, 1000, 1000), activation tanh, network score: ' + str(score4) + ', mean squared error: ' + str(mse4))

