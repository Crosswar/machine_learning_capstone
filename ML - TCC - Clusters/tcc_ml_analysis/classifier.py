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
                                                    test_size = 0.2,
                                                    random_state = 0)

# Exploratory Data - Export results to excel or images
exportData(data, y_train, y_test, label)
exportPlots(data, False)
exportMetrics()

### Logistic Regression
model_lr = LogisticRegression(random_state=7, penalty='l1')

clf_lr = model_lr.fit(X_train, y_train)

pred_lr = model_lr.predict(X_test)
print(classification_report(y_test, pred_lr))
print(clf_lr.score(X_test, y_test))

pred = model_lr.predict(features)
print(clf_lr.score(features, label))

sns.set(font_scale=1.8)
#fig = plt.figure(figsize=(15, 9));

ab = data.copy()
ab['PA'] = ab['restbp'].apply(lambda x: 'Normal' if x < 138 else 'Alta')
ab['th'] = 220
ab['100%'] = ab[['th']].sub(ab['age'], axis=0)
ab['NF'] = ab[['100%']].sub(ab['thalach'], axis=0)
ab['Freq.'] = ab['NF'].apply(lambda x: 'Normal' if x >= 0 else 'Anormal')
ab['Idade'] = ab['age']
ab['PA Repouso'] = ab['restbp']
ab['sex'] = ab['sex'].apply(lambda x: 'Mulher' if x == 0 else 'Homem')
#ab['fbs'] = ab['fbs'].apply(lambda x: '> 120 mg/dl' if x == 1 else '< 120mg/dl')
#g = sns.catplot(x="num", hue="fbs", col="sex", data=ab, kind="count", height=4, aspect=.7);

#ax = sns.relplot(x="Idade", y="PA Repouso", hue="PA", data=ab, s=100, height=12);
#plt.show();


#cm = confusion_matrix(y_test, pred_lr);
#ax = sns.heatmap(cm,annot=True, cmap='Blues', fmt='g')
#ax.set_xlabel(xlabel='DCV', fontsize=16)

#ax.set_title(label='Matriz de Confusão', fontsize=20)
# New Matrix
#cols = ['idade', 'sexo', 'cp', 'p.a', 'col', 'glicemia', 'ecg', 'freq.', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'dcv']
#a = data.copy()
#a.columns = cols
##f=["age", "restbp", "chol", "thalach", "oldpeak", "num"]
#corr = a.corr()
#mask = np.zeros_like(corr)
#mask[np.triu_indices_from(mask)] = True
#with sns.axes_style("white"):
#    ax = sns.heatmap(corr, mask=mask, square=False, annot=True, cmap='coolwarm_r')

#g = sns.lmplot(x="age", y="num", data=pred_lr, logx=True)





