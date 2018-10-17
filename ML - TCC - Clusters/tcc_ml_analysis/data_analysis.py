# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:39:06 2018

@author: strin
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set dpi of images to obtain a higher quality
mpl.rc("savefig", dpi=150)
sns.set(color_codes=True)

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

# Export Matrices and Regression Plots
def exportPlots(data, flag):
    if(flag):
        # Calculate correlation
        corr = data.corr()

        # Plot correlation matrix
        fig = plt.figure(figsize=(15, 9))
        mask = np.zeros_like(corr, dtype=np.bool) # create mask to cover the upper triangle
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(corr, annot=True, mask=mask, linewidths=0.01)

        fig.suptitle('Correlation Matrix', fontsize=14)
        fig.savefig('plots/matrices/correlation_matrix.png')

        #sns.lmplot(x="age", y="chol", hue="num", col="fbs", row="sex", data=data)
        sns.jointplot(x="age", y="chol", hue="num", data=data, kind="reg");


# Export Categorical
def exportCategorial():
    return True

def exportMetrics():
    return True