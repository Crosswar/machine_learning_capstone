#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:09:56 2018
@author: gabrielhenriques
"""

import numpy as np
import pandas as pd
from utils import fit_model, run_dendogram, get_data
from visuals import get_plot

#import dataset
dataset = pd.read_csv('cleveland.csv')

# Allocate Columns
# Prediction: 0 - Sex, 4 - Cholesterol
# Label:      1 - Gender
X = dataset.iloc[:, [0, 4, 1]].values

# Dendogram Analysis - Cluster Number Decision Tool
# Expects: Data Array, Method, Run(Bool)
run_dendogram(X, 'ward', False)

# Fit model
# Expects: Number of Clusters, Affinity, Linkage, Data Array
predict = fit_model(4, 'euclidean', 'ward', X)

# Get Data - Returns x,y axis and label
# Expects: Data Array, Predict Result, Cluster Index
c_idx = 0
data_matrix = []

while c_idx <= max(predict):
    data_matrix.append(get_data(X, predict, c_idx))
    c_idx += 1


# Data Visualization
# Expects: DataMatrix 0 - 1 - DisplaySize - 2 (Color Attribute)
plt_idx = 0

while plt_idx < len(data_matrix):
    plotInfo = ['Cluster ' + str(plt_idx + 1), 'Age', 'Cholesterol']
    get_plot(data_matrix[plt_idx][0], data_matrix[plt_idx][1], 60, data_matrix[plt_idx][2], plotInfo)
    plt_idx += 1

#print(np.mean(data_matrix[1]))
#print(np.std(data_matrix[1]))

