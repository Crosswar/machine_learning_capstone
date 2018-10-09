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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# Read CSV and rename columns
columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.read_csv('cleveland.csv', header = 0, names = columns)

# Encoding to a binary value
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Features
features = df.drop('num', axis = 1)

# Predict Attribute (label) - Heart Disease
hd = df['num']

# Split the 'features' and the label data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    hd,
                                                    test_size = 0.1,
                                                    random_state = 6)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv= 3, random_state= 0,
                           multi_class='multinomial').fit(X_train, y_train)
predictions = clf.predict(X_test)

print(clf.score(features, hd))
print(classification_report(y_test, predictions))


# Example of a confusion matrix in Python

classes = ['Positive HD', 'False HD']

results = confusion_matrix(y_test, predictions)
plt.figure(figsize = (10,7))

f = sns.heatmap(results, annot=True)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show(f)
# Correlation and Pairplot
#fig = plt.figure(figsize=(20, 20))
#ax = plt.gca()
# sns.heatmap(df.corr(), annot=True)
# sns.pairplot(df)