from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


data_unnorm, result = make_moons(n_samples=1000, noise=0.05, random_state=100)
scaler = MinMaxScaler()
data = scaler.fit_transform(data_unnorm)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[:, 0], data[:, 1], c = result)
plt.show()

model_kmeans = KMeans(n_clusters=2)
model_kmeans.fit(data)
print("Sihouette Score KMeans: %0.3f" % metrics.silhouette_score(data, model_kmeans.labels_))
print("DB Score KMeans: %0.3f" % metrics.davies_bouldin_score(data, model_kmeans.labels_))
print("Rand Score KMeans: %0.3f" % metrics.rand_score(result, model_kmeans.labels_))
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[:, 0], data[:, 1], c = model_kmeans.labels_)
plt.show()

model_AC = AgglomerativeClustering(n_clusters=2)
model_AC.fit(data)
print("Sihouette Score AgglomerativeClustering: %0.3f" % metrics.silhouette_score(data, model_AC.labels_))
print("DB Score AgglomerativeClustering: %0.3f" % metrics.davies_bouldin_score(data, model_AC.labels_))
print("Rand Score KMeans: %0.3f" % metrics.rand_score(result, model_AC.labels_))
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[:, 0], data[:, 1], c = model_AC.labels_)
plt.show()

model_DB = DBSCAN(eps=0.1, min_samples=5)
model_DB.fit(data)
labels = model_DB.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Sihouette Score DBSCAN: %0.3f" % metrics.silhouette_score(data, model_DB.labels_))
print("DB Score DBSCAN: %0.3f" % metrics.davies_bouldin_score(data, model_DB.labels_))
print("Rand Score KMeans: %0.3f" % metrics.rand_score(result, model_DB.labels_))
print('Liczba klastrów wytworzona przez algorytm w procesie uczenia: %d' % n_clusters_)
print('Liczba wskazanych przez algorytm punktów szumu: %d' % n_noise_)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[:, 0], data[:, 1], c = model_DB.labels_)
plt.show()

input('Click Enter to continue...')