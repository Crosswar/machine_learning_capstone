# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:39:06 2018

@author: strin
"""
import pandas as pd

def exportData(data, y_train, y_test, label):

    exploreDF = data.copy()
    exploreDF.drop(['cp', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],
                   axis=1,
                   inplace=True)

    data_description = exploreDF.describe()
    data_samples = data.head()

    freqs = pd.DataFrame({ "Training dataset": y_train.value_counts().tolist(),
                           "Test dataset":y_test.value_counts().tolist(),
                           "Total": label.value_counts().tolist()}, index=["Healthy", "Sick"])
    freqs[["Training dataset", "Test dataset", "Total"]]

    # Export to Excel - Separate Sheets
    writer = pd.ExcelWriter('exploratory_analysis.xlsx')

    data_description.to_excel(writer,'Data Description')
    data_samples.to_excel(writer,'Data Samples')
    freqs.to_excel(writer,'Classes Balance')

    writer.save()

def exportCategorial():
    return True

def exportMetrics():
    return True