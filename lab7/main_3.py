import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import skfuzzy as fuzz
dataUnnorm = []

with open('planets.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        dataUnnorm.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])

scaler = MinMaxScaler()
data = scaler.fit_transform(dataUnnorm)
cntr, u_orig, u0, d, jm, p, fpc = fuzz._cluster.cmeans(data, 5, 2, error=0.005, maxiter=1000)
print(np.array2string(cntr, max_line_width=np.inf))

input('Click Enter to continue...')