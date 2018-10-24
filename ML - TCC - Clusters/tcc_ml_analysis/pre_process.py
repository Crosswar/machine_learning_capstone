#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 02:37:12 2018

@author: gabrielhenriques
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def readData():
    # Read CSV and rename columns
    columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = pd.read_csv('cleveland.csv', header = 0, names = columns)

    #df2 = df.drop(['chol', 'age', 'restbp', 'thalach'], axis = 1)

    # Scale Data
    #scaler = StandardScaler()
    #c = scaler.fit_transform(df[['chol', 'age', 'restbp', 'thalach']])
    #d = pd.DataFrame(c, columns=['chol', 'age', 'restbp', 'thalach'])

    return df

def cleanData(df):
    # Encoding heart-disease class to a binary value
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    return df

def getFeatures(df):
    # Features
    features = df.drop('num', axis = 1)

    # Predict Attribute (label) - Heart Disease
    label = df['num']
    return features, label

def prepareDataForCharts(data):
    data2 = data.copy()

    # Create new columns with proper names for visualization
    data2['Colesterol'] = data2['chol']
    data2['Idade'] = data2['age']
    data2['Freq. Cardíaca'] = data2['thalach']

    data2['sex'] = data2['sex'].apply(lambda x: 'Mulher' if x == 0 else 'Homem')
    data2['DCV'] = data2['num'].apply(lambda x: 'Sim' if x == 1 else 'Não')
    data2['Glicemia Jejum'] = data2['fbs'].apply(lambda x: '> 120 mg/dl' if x == 1 else '< 120mg/dl')

    return data2

def prepareDataForExcel(data):
    exploreDF = data.copy()
    exploreDF.drop(['cp', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],
                   axis=1,
                   inplace=True)
    return exploreDF