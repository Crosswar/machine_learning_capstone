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

# Specific Files of the project
from pre_process import readData, cleanData, getFeatures
from data_analysis import exportData, exportMetrics
from visual_analysis import exportPlots


# Get the data, split labels and retrieve the features
data = readData()
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

data2 = data.copy()

# Create new columns with proper names for visualization
data2['Colesterol'] = data2['chol']
data2['Idade'] = data2['age']
data2['Freq. Cardíaca'] = data2['thalach']

data2['Sexo'] = data2['sex'].apply(lambda x: 'Mulher' if x == 0 else 'Homem')
data2['DCV'] = data2['num'].apply(lambda x: 'Sim' if x == 1 else 'Não')
data2['Glicemia Jejum'] = data2['fbs'].apply(lambda x: '> 120 mg/dl' if x == 1 else '< 120mg/dl')

sns.set(color_codes=True)
fig = plt.figure(figsize=(15, 9))
sns.set(font_scale=1.5)

sns.scatterplot(x="Idade", y="Colesterol", hue="DCV", style="DCV", data=data2, s=100)
plt.axvline(55, color="k", linestyle="--", linewidth=1)
plt.axhline(200, color="k", linestyle="--", linewidth=1)
# Visual Analysis
# Preprocessing TODO
#min_max_scaler = MinMaxScaler()

#train_scaled = min_max_scaler.fit_transform(X_train)
#test_scaled = min_max_scaler.fit_transform(X_test)

#df_normalized = pd.DataFrame(train_scaled)
#df_normalized2 = pd.DataFrame(test_scaled)




# Example of a confusion matrix in Python
#results = confusion_matrix(y_test, predictions)
#plt.figure(figsize = (10,7))

#f = sns.heatmap(cm, annot=True)

#plt.xlabel("Predição")
#plt.ylabel("Valor Real")
#plt.show(f)
# Correlation and Pairplot
#fig = plt.figure(figsize=(20, 20))
#ax = plt.gca()
# sns.heatmap(df.corr(), annot=True)
# sns.pairplot(df)


# Standardize
#scaler = StandardScaler()
#c = scaler.fit_transform(df[['chol', 'age']])
#d = pd.DataFrame(c, columns=['chol', 'age'])
#sns.distplot(d['chol'])
#x_array = np.array(df['age'])
#xx = normalize([x_array])

#scaler = StandardScaler()
#Xa = scaler.fit_transform(features)