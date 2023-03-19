import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

digits = load_digits()
target = digits.target
data = digits.data[:, :64]
x3, x3_test, y3, y3_test = train_test_split(data, target, test_size=0.2)
for activation_type in ['identity', 'tanh', 'relu']:
    for max_iter in [100, 500, 2000]:
        for solver in ['lbfgs', 'adam', 'sgd']:
            if solver == 'sgd':
                for learning_rate in ["constant", "invscaling", "adaptive"]:
                    model = MLPClassifier(hidden_layer_sizes=(10,5), activation=activation_type,
                    max_iter=max_iter, solver=solver, learning_rate=learning_rate)
                    model.fit(x3, y3)
                    print('Activation function: ' + activation_type + ', max_iter: ' + str(max_iter) + ', solver: ' + solver + ', learning rate: ' + learning_rate)
                    print('Accuracy:', model.score(x3_test,y3_test))
                    if model.score(x3_test,y3_test) < 0.1 or model.score(x3_test,y3_test) > 0.93:
                        predicted_labels = model.predict(x3)
                        matrix = confusion_matrix(y3, predicted_labels)
                        print(matrix)
            else:
                model = MLPClassifier(hidden_layer_sizes=(10,5), activation=activation_type,max_iter=max_iter, solver=solver)
                model.fit(x3, y3)
                print('Activation function: ' + activation_type + ', max_iter: ' + str(max_iter) + ', solver: ' + solver)
                print('Accuracy:', model.score(x3_test,y3_test))
                if model.score(x3_test,y3_test) < 0.1 or model.score(x3_test,y3_test) > 0.93:
                        predicted_labels = model.predict(x3)
                        matrix = confusion_matrix(y3, predicted_labels)
                        print(matrix)