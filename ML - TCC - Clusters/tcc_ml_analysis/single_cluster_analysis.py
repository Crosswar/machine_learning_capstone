#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:09:56 2018
@author: gabrielhenriques
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn import mixture

#import dataset
dataset = pd.read_csv('cleveland.csv')
X = dataset.iloc[:, [0, 4, 1]].values

# Fit model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(
    n_clusters = 4, affinity = 'euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Plot
plot.scatter(X[:, 0], X[:, 1], s = 100, c = y_hc)

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
'''

gmm = mixture.GaussianMixture(n_components=4)
gmm.fit(X)
y_hc = gmm.predict(X)
plot.scatter(X[:, 0], X[:, 1], s = 100, c = y_hc, label = 'Cluster 1')'''
