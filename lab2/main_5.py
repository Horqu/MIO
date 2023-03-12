import random
import sklearn
from sklearn.linear_model import Perceptron
from sklearn import datasets

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
    print('Accuracy for ' + str(percent) + '%/' + str(100 - percent) + '% training/testing data with ' + str(iterations) + ' iterations: ' + str(neuron.score(x_test, y_test)))

iris_test(50, 2)
iris_test(50, 3)
iris_test(50, 5)
iris_test(50, 10)
iris_test(50, 20)
iris_test(50, 50)
