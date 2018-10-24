# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:26:01 2018

@author: gabriel_henriques
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from pre_process import prepareDataForCharts

# Fixed variables
data_names = ['Idade', 'Colesterol', 'Freq. Cardíaca']
data_columns = ['age', 'chol', 'thalach']

# Set dpi of images to obtain a higher quality
mpl.rc("savefig", dpi=150)
sns.set(color_codes=True)

# Define a folder to save images
def setPlotFolder(name):
    return 'images/Plot_'+ name + '.png'

# Export Matrices and Regression Plots
def exportPlots(data, flag):
    if(flag):
        # Create Labels and define values for better visualization
        prep_data = prepareDataForCharts(data)


        # Generate Specific plots
        exportCorrelationMatrix(data)
        exportAttributeAnalysis(prep_data)
        exportDataExploration(data)

def exportAttributeAnalysis(data):
    fig = plt.figure(figsize=(15, 9))
    # Thalach Attribute vs Age with respect to Heart Disease
    fig = sns.relplot(x="Idade", y="Freq. Cardíaca", kind="line", data=data, ci=None, hue="DCV", aspect=2)
    fig.savefig(setPlotFolder("Freq_Idade"))

    # Scatter to explore dataset
    #sns.scatterplot(x="Idade", y="Colesterol", hue="DCV", style="DCV", data=data, s=100)
    #fig.savefig("images/Scatter.png")


def exportCorrelationMatrix(data):
        # Calculate correlation
        corr = data.corr()

        # Plot correlation matrix
        mask = np.zeros_like(corr, dtype=np.bool) # create mask to cover the upper triangle
        mask[np.triu_indices_from(mask)] = True

        fig = plt.figure(figsize=(15, 9))
        sns.heatmap(corr, annot=True, mask=mask, linewidths=0.01)

        fig.suptitle('Matriz de Correlação', fontsize=14)
        fig.savefig(setPlotFolder("matriz_correlação"))

def exportDataExploration(data):
        sns.set(font_scale=2)
        # Generate a chart for each column specified on the constant data_columns
        for i, z in zip(data_names, data_columns):
            fig = plt.figure(figsize=(15, 9))
            sns.distplot(data[z], fit=norm, axlabel=i, kde=False, color='blue')
            fig.savefig(setPlotFolder(i))

        # Regression Plot
        #sns.lmplot(x="age", y="chol", hue="num", col="fbs", row="sex", data=data)
        #sns.jointplot(x="age", y="chol", hue="num", data=data, kind="reg")

        ##sns.distplot(data[att], fit=norm, axlabel=label, kde=False, color='blue')
        #sns.distplot(data['chol'], fit=norm, axlabel='Colesterol', rug = True, bins=15, kde=False, color='blue')