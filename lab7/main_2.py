import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
dataUnnorm = []

with open('planets.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        dataUnnorm.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9])])

scaler = MinMaxScaler()
data = scaler.fit_transform(dataUnnorm)

model_kmeans = KMeans(n_clusters=5)
model_kmeans.fit(data)
print("Sihouette Score KMeans: %0.3f" % metrics.silhouette_score(data, model_kmeans.labels_))
print("DB Score KMeans: %0.3f" % metrics.davies_bouldin_score(data, model_kmeans.labels_))
print(np.array2string(model_kmeans.cluster_centers_, max_line_width=np.inf))

model_AC = AgglomerativeClustering(n_clusters=5)
model_AC.fit(data)
print("Sihouette Score AgglomerativeClustering: %0.3f" % metrics.silhouette_score(data, model_AC.labels_))
print("DB Score AgglomerativeClustering: %0.3f" % metrics.davies_bouldin_score(data, model_AC.labels_))

model_DB = DBSCAN(eps=0.5, min_samples=5)
model_DB.fit(data)
labels = model_DB.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Sihouette Score DBSCAN: %0.3f" % metrics.silhouette_score(data, model_DB.labels_))
print("DB Score DBSCAN: %0.3f" % metrics.davies_bouldin_score(data, model_DB.labels_))
print('Liczba klastrów wytworzona przez algorytm w procesie uczenia: %d' % n_clusters_)
print('Liczba wskazanych przez algorytm punktów szumu: %d' % n_noise_)


input('Click Enter to continue...')