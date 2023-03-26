import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x = np.arange(-2 * np.pi, 2 * np.pi, 0.05)
y = np.sin(x)

X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y)
network = MLPRegressor(solver='adam', hidden_layer_sizes=(20), max_iter=1000, tol = 0.001, activation='tanh')
network.fit(X_train, y_train)
y_pred = network.predict(x.reshape(-1, 1))
print('coefs ' + str(network.coefs_))
print('intercepts ' + str(network.intercepts_))
plt.plot(x, y, 'b')
plt.plot(x, y_pred, 'r')
plt.show()