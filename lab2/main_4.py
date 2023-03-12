import random
import sklearn
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn import datasets

iris = sklearn.datasets.load_iris()

def iris_test(percent):
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

    neuron = Perceptron(tol = 1e-3, max_iter = 20)
    neuron.fit(x, y)
    print('Testing for ' + str(percent) + '%/' + str(100 - percent) + '% training/testing data')
    print('Accuracy: ' + str(neuron.score(x_test, y_test)))

    y_predicted = neuron.predict(x_test)
    confusion_matrix_model = confusion_matrix(y_test, y_predicted)
    print('Confusion matrix:')
    print(confusion_matrix_model)

iris_test(10)
iris_test(50)
iris_test(90)
