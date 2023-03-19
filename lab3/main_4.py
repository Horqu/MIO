import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

yeast = np.loadtxt('yeast.data', dtype='str')[:, 1:]
labels = np.reshape(np.unique(yeast[:, 8]), (10,1))
indexes = np.reshape(np.array(range(0, len(labels))), (10,1))
labels = np.append(labels, indexes, axis=1)
for l in labels:
    yeast[yeast[:, 8] == l[0], 8] = l[1]
yeast = np.array(yeast, dtype='float')
x4, x4_test, y4, y4_test = train_test_split(yeast[:,0:7], yeast[:,8], test_size=0.1, stratify=yeast[:,8])
model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=2000, activation='relu', solver='adam')
model.fit(x4, y4)
print('Accuracy:', model.score(x4_test,y4_test))
y_predicted = model.predict(x4_test)
confusion_matrix_model = confusion_matrix(y4_test, y_predicted)
print('Confusion matrix')
print(confusion_matrix_model)
print('Balanced accuracy:', balanced_accuracy_score(y4_test, model.predict(x4_test)))