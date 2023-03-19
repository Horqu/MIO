from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

digits = sklearn.datasets.load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits['data'], digits['target'], stratify = digits['target'], test_size=0.2)

model = MLPClassifier(max_iter = 2000)
model.fit(x_train, y_train)
print('Accuracy: ', model.score(x_test, y_test))