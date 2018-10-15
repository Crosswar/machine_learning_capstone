# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:39:06 2018

@author: strin
"""
import pandas as pd

def exportData(data):

    # Rename Columns to generate Excel tables
    new_cols = ['Age', 'Sex', 'ChestPain', 'RestingBP', 'Cholesterol', 'FastingSugar', 'RestECG', 'Thalach', 'ExerciseInducedAngina']
    data.columns = new_cols

    exploreDF = data.copy()
    exploreDF.drop(['cp', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],
                   axis=1,
                   inplace=True)

    data_description = exploreDF.describe()
    data_samples = data.head()

    # Export to Excel - Separate Sheets
    writer = pd.ExcelWriter('exploratory_analysis.xlsx')
    data_description.to_excel(writer,'Data Description')
    data_samples.to_excel(writer,'Data Samples')
    writer.save()