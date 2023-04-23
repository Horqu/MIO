import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

with open('press_readers_chicago.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    data = []
    for row in csv_reader:
        parts = row[0].split(';')
        data.append([int(parts[0]), int(parts[1])])

raw_data = np.array(data)
rd1, rd2 = np.transpose(raw_data)
kmean = KMeans(n_clusters=8, random_state=0).fit(raw_data)
print("Sihouette Score bez normalizacji: %0.3f" % metrics.silhouette_score(raw_data, kmean.labels_))
print("DB Score bez normalizacji: %0.3f" % metrics.davies_bouldin_score(raw_data, kmean.labels_))
# print("Rand Score bez normalizacji: %0.3f" % metrics.rand_score(raw_data, kmean.labels_))
f, ax1 = plt.subplots(1, 1, sharey=True,figsize=(10, 6))
ax1.set_title('Etykiety nadane przez k-means: ')
ax1.scatter(rd1, rd2,c=kmean.labels_)
plt.show()

scaler = MinMaxScaler()
raw_data_norm = scaler.fit_transform(raw_data)
rd3, rd4 = np.transpose(raw_data_norm)
kmean2 = KMeans(n_clusters=4, random_state=0).fit(raw_data_norm)

print("Sihouette Score z normalizacja: %0.3f" % metrics.silhouette_score(raw_data_norm, kmean2.labels_))
print("DB Score z normalizacja: %0.3f" % metrics.davies_bouldin_score(raw_data_norm, kmean2.labels_))
# print("Rand Score z normalizacja: %0.3f" % metrics.rand_score(raw_data_norm, kmean2.labels_))
f, ax2 = plt.subplots(1, 1, sharey=True,figsize=(10, 6))
ax2.set_title('Etykiety nadane przez k-means: ')
ax2.scatter(rd3, rd4, c=kmean2.labels_)
plt.show()
input('Click Enter to continue...')