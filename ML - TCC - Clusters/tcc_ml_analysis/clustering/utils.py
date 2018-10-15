#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:09:56 2018
@author: gabrielhenriques
"""
import numpy as np

# Dendogram Analysis
def run_dendogram(data, met, run):
    
    if run:
        # Import Hierarchy Cluster Tool
        import scipy.cluster.hierarchy as sch
        
        # Generate Dendogram Plot
        dendogram = sch.dendrogram(sch.linkage(data, method = met))
        return dendogram

# Fit model
def fit_model(n_c, affinity, linkage, data): 
    
    # Import AgglomerativeClustering
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(
        n_clusters = n_c, affinity = affinity, linkage = linkage)
    
    # Fit Data - Expecting: arr[:, [x,y]]
    predict = hc.fit_predict(data)
    return predict

# Create Data Matrix and return cluster variables
def get_data(data, predict, c_idx):
    
    # Define x,y axis based on which cluster index is received
    x = np.int_(data[predict == c_idx, 0]).tolist()
    y = np.int_(data[predict == c_idx, 1]).tolist()
    
    # Define label as third param from the received data array
    label = np.int_(data[predict == c_idx, 2]).tolist()
    
    # Build data_matrix with (x,y,label)
    data_matrix = [x, y, label]
    return data_matrix