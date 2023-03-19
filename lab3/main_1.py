import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

def train(layer, iterations, activation_type):
    model = MLPClassifier(hidden_layer_sizes = layer, max_iter = iterations, activation = activation_type)
    model.fit(x_train, y_train)
    print(activation_type + ', layers: ' + str(layer))
    print('Accuracy score: ' + str(model.score(x_test, y_test)))
    display = DecisionBoundaryDisplay.from_estimator(model, x_test, response_method = 'predict', alpha = 0.5)
    display.ax_.scatter(x_test[:, 0], x_test[:, 1], c = y_test, edgecolor= 'k')
    plt.show()

data = np.loadtxt('treatment.txt')
x = data[:, 0:2]
y = data[:, 2:3]
x[:, 0] /= np.max(x[:, 0])
x[:, 1] /= np.max(x[:, 1])
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.2)

train((3, 3), 2000, 'relu')
train((5, 5), 2000, 'relu')
train((3, 3), 2000, 'tanh')
train((5, 5), 2000, 'tanh')
train((3, 3), 2000, 'identity')
train((5, 5), 2000, 'identity')
