#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:09:56 2018
@author: gabrielhenriques
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

#import dataset
dataset = pd.read_csv('cleveland.csv')
X = dataset.iloc[:, [0, 4, 1]].values

# Fit model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(
    n_clusters = 4, affinity = 'euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

plot.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plot.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plot.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plot.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'black', label = 'Cluster 4')

# Plot Info
plot.title('Clusters')

# Axis Labels
plot.xlabel('Age')
plot.ylabel('Cholesterol')

plot.show()

# Mean, Variance and Standard Deviation for Each Cluster
c_out = []
mean_list = []
deviation_list = []

for c_index in range(max(y_hc) + 1):
    c_out.append(np.sort(X[y_hc == c_index, 1].astype(int), axis = 0))
    final_out = c_out[c_index]
    
    mean_list.append(np.mean(final_out))
    #mean_list.append(c_index)
    deviation_list.append(np.std(final_out))
    #deviation_list.append(c_index)

# Print Options

# Print Mean
print(mean_list)

# Print STD Deviation
print(deviation_list)
