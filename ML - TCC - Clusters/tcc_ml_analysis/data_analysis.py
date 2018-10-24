# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:39:06 2018

@author: gabriel_henriques
"""
import pandas as pd
import numpy as np
from pre_process import prepareDataForExcel

def exportData(data, y_train, y_test, label):
    exploreDF = prepareDataForExcel(data)

    data_description = exploreDF.describe()
    data_samples = data.head()

    freqs = pd.DataFrame({ "Training dataset": y_train.value_counts().tolist(),
                           "Test dataset":y_test.value_counts().tolist(),
                           "Total": label.value_counts().tolist()}, index=["Healthy", "Sick"])
    freqs[["Training dataset", "Test dataset", "Total"]]

    # Intrinsic Discrepancy
    intr_discr, cols = calculateDiscrepancy(data)
    intr_list = [cols, intr_discr]
    r_discr = pd.DataFrame.from_dict(intr_list)

    # Export to Excel - Separate Sheets
    writer = pd.ExcelWriter('exploratory_analysis.xlsx')

    data_description.to_excel(writer,'Data Description')
    data_samples.to_excel(writer,'Data Samples')
    freqs.to_excel(writer,'Classes Balance')
    r_discr.to_excel(writer, 'Intrinsic Discrepancy')

    writer.save()


def exportMetrics():
    return True

def intrinsicDiscrepancy(x, y):
    sumx = sum(xval for xval in x)
    sumy = sum(yval for yval in y)
    id1  = 0.0
    id2  = 0.0
    for (xval,yval) in zip(x,y):
        if (xval>0) and (yval>0):
            id1 += (float(xval)/sumx) * np.log((float(xval)/sumx)/(float(yval)/sumy))
            id2 += (float(yval)/sumy) * np.log((float(yval)/sumy)/(float(xval)/sumx))
    return min(id1, id2)

def calculateDiscrepancy(data):
    discr_list = []
    discr_list2 = []

    for col in range(len(data.columns)):
        new_data = data.iloc[:, col]

        hist, bin_edges   = np.histogram(new_data, density=False)
        hist1, bin_edges1 = np.histogram(new_data[data.num > 0], bins=bin_edges, density=False)
        hist2, bin_edges2 = np.histogram(new_data[data.num == 0], bins=bin_edges, density=False)

        colum_name = data.columns.values[col]
        discrepancy = intrinsicDiscrepancy(hist1, hist2)

        discr_list2.append(str(colum_name))
        discr_list.append(str(round(discrepancy, 3)))
    return discr_list, discr_list2
