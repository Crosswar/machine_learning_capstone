#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 02:37:12 2018

@author: gabrielhenriques
"""

import pandas as pd

def readData():
    # Read CSV and rename columns
    columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = pd.read_csv('cleveland.csv', header = 0, names = columns)
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