from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

def trained_neuron(number_of_points, x_test_data, y_test_data):
    x_data = np.concatenate((np.random.normal([0, -1], [1, 1], [number_of_points, 2]), np.random.normal([1, 1], [1, 1], [number_of_points, 2])))
    y_data = np.concatenate((np.array([0] * number_of_points), np.array([1] * number_of_points)))
    neuron = Perceptron(tol = 1e-3, max_iter = 20)
    neuron.fit(x_data, y_data)
    draw_hyperplane(neuron, x_data, y_data, x_test_data, y_test_data)
    

def draw_hyperplane(neuron, x_data, y_data, x_test_data, y_test_data):
    print('Hyperplane equation: ' + str(neuron.coef_[0][0]) + ' * x_1 + ' + str(neuron.coef_[0][1]) + ' * x_2 + ' + str(neuron.intercept_[0]) + ' = 0')
    print('Test data accuracy: ' + str(neuron.score(x_test_data, y_test_data)))
    x1 = np.linspace(-4, 4, 100)
    x2 = -(1./neuron.coef_[0][1]) * (neuron.coef_[0][0] * x1 + neuron.intercept_[0]) 
    plt.plot(x1, x2, '-r')
    x_cumulative = np.concatenate((x_data, x_test_data))
    y_test_data = [x + 2 for x in y_test_data]
    y_cumulative = np.concatenate((y_data, y_test_data))
    plt.scatter(np.array(x_cumulative)[:, 0], np.array(x_cumulative)[:, 1], c = y_cumulative)
    plt.show()

x_test_data = np.concatenate((np.random.normal([0, -1], [1, 1], [200, 2]), np.random.normal([1, 1], [1, 1], [200, 2])))
y_test_data = np.concatenate((np.array([0] * 200), np.array([1] * 200)))

# trained_neuron(5, x_test_data, y_test_data)
# trained_neuron(10, x_test_data, y_test_data)
# trained_neuron(20, x_test_data, y_test_data)
trained_neuron(100, x_test_data, y_test_data)