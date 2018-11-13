# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:19:56 2018

@author: Gabriel
"""
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

import statsmodels.api as sm

# Specific Files of the project
from pre_process import readData, cleanData, getFeatures, prepareDataForCharts
from data_analysis import exportData, exportMetrics
from visual_analysis import exportPlots

# Get the data, split labels and retrieve the features
data, columns2 = readData()
cleanData(data)

features, label = getFeatures(data)

# Split the 'features' and the label data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    label,
                                                    test_size = 0.3,
                                                    random_state = 0)

# Exploratory Data - Export results to excel or images
exportData(data, y_train, y_test, label)
exportPlots(data, False)
exportMetrics()


#min_max_scaler = MinMaxScaler()

#train_scaled = min_max_scaler.fit_transform(X_train)
#test_scaled = min_max_scaler.fit_transform(X_test)

#df_normalized = pd.DataFrame(train_scaled)
#df_normalized2 = pd.DataFrame(test_scaled)








