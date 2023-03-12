from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

fuel = np.loadtxt('fuel.txt')
X = fuel[:, 0:3]
Y = fuel[:, 3:4]
neuron = Perceptron(tol = 1e-3, max_iter = 5)
neuron.fit(X, Y)
print('Accuracy: ' + str(neuron.score(X, Y)))
