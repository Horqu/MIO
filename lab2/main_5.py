import random
import sklearn
from sklearn.linear_model import Perceptron
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = sklearn.datasets.load_iris()

def iris_test(percent, iterations):
    x = []
    y = []
    x_test = []
    y_test = []

    for i in range(0, iris['target'].size):
        if random.randint(1, 100) <= percent:
            x.append(iris['data'][i])
            y.append(iris['target'][i])
        else:
            x_test.append(iris['data'][i])
            y_test.append(iris['target'][i])

    neuron = Perceptron(early_stopping = False, max_iter = iterations)
    neuron.fit(x, y)
    # print('Accuracy for ' + str(percent) + '%/' + str(100 - percent) + '% training/testing data with ' + str(iterations) + ' iterations: ' + str(neuron.score(x_test, y_test)))
    return neuron.score(x_test, y_test)

accuracy_array = [0, 0, 0, 0, 0, 0, 0]
for i in range(20):
    accuracy_array[0] += iris_test(70, 1)
    accuracy_array[1] += iris_test(70, 2)
    accuracy_array[2] += iris_test(70, 3)
    accuracy_array[3] += iris_test(70, 5)
    accuracy_array[4] += iris_test(70, 10)
    accuracy_array[5] += iris_test(70, 20)
    accuracy_array[6] += iris_test(70, 50)

for i in range(len(accuracy_array)):
    accuracy_array[i] /= 20

plt.scatter([1, 2, 3, 5, 10, 20, 50], accuracy_array)
plt.xlabel('Number of iterations')
plt.ylabel('Average accuracy')
plt.show()