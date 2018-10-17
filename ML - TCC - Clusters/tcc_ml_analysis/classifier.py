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
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

from pre_process import readData, cleanData, getFeatures
from data_analysis import exportData, exportMetrics, exportPlots

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

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




### Classifiers
### Logistic Regression

model_lr = LogisticRegression(random_state=7)

clf_lr = model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)

print(classification_report(y_test, pred_lr))
print(clf_lr.score(X_test, y_test))


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
svc = LogisticRegression()
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),
              scoring='accuracy')
rfecv.fit(X_test, y_test)

print("Optimal number of features : %d" % rfecv.n_features_)
print(rfecv.get_support(indices=True))
X_new = rfecv.transform(features)
print(features.columns[rfecv.get_support()])
#print(np.absolute(rfecv.estimator_.coef_))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


### Logistic Regression - CrossValidation
clf_lrCV = LogisticRegressionCV(cv= 3, random_state= 0, penalty='l2',
                           multi_class='multinomial').fit(X_train, y_train)

# Predict class labels for the training set
predicted2 = clf_lrCV.predict(X_test)

print('Mean Absolute Error LRCV- ', mean_absolute_error(y_test, predicted2))
print('LogisticRegression Test - ', clf_lrCV.score(X_test, y_test))

### Gaussian Naive Bayes
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
predicted3 = gnb.predict(X_test)
print('Mean Absolute Error GNB - ', mean_absolute_error(y_test, predicted3))
print('Gaussian NB Test - ', gnb.score(X_test, y_test))

#cm = confusion_matrix(y_train, predicted1)
print(classification_report(y_test, predicted2))
print(classification_report(y_test, predicted3))


#gnb_lr=VotingClassifier(estimators=[('Guassian Naive Bayes', gnb),('Logistic Regression', clf_lr)], voting='soft', weights=[2,1]).fit(X_train,y_train)
#print('The accuracy for Guassian Naive Bayes and Logistic Regression:',gnb_lr.score(X_test,y_test))

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