# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:19:56 2018

@author: Gabriel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.naive_bayes import GaussianNB

from pre_process import readData, cleanData, getFeatures
from data_analysis import exportData

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


# Get the data, split labels and retrieve the features
data = readData()
cleanData(data)
features, label = getFeatures(data)


# Exploratory Data - Export results to excel
exportData(data)



# Split the 'features' and the label data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    label,
                                                    test_size = 0.3,
                                                    random_state = 43)


# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


model = LogisticRegression(fit_intercept=True,penalty='l1',dual=False,C=1000.0)
a = model.fit(features, label)
ypred = model.predict(X_test)

print(classification_report(y_test, ypred))
print(a.score(X_test, y_test))

clf = LogisticRegressionCV(cv= 3, random_state= 0, penalty='l2',
                           multi_class='multinomial').fit(X_train, y_train)

# predict class labels for the training set
predicted2 = clf.predict(X_test)

print('Mean Absolute Error - ', mean_absolute_error(y_test, predicted2))
print('LogisticRegression Test - ', clf.score(X_test, y_test))

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
predicted3 = gnb.predict(X_test)
print('Gaussian NB Test - ', gnb.score(X_test, y_test))

#cm = confusion_matrix(y_train, predicted1)
print(classification_report(y_test, predicted2))
print(classification_report(y_test, predicted3))

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



#sns.distplot(df['age'])

# Standardize
#scaler = StandardScaler()
#c = scaler.fit_transform(df[['chol', 'age']])
#d = pd.DataFrame(c, columns=['chol', 'age'])
#sns.distplot(d['chol'])
#x_array = np.array(df['age'])
#xx = normalize([x_array])

#scaler = StandardScaler()
#Xa = scaler.fit_transform(features)