#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 00:09:56 2018
@author: gabrielhenriques
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Plot Config/Info
def plotConfig(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()
    return ax, fig
    
# Define label color and name for attribute DataMatrix[2]
def plotLabels():
    red_patch = mpatches.Patch(color ='red', label='Male')
    blue_patch = mpatches.Patch(color ='blue', label='Female')
    legend = plt.legend(handles = [red_patch, blue_patch])
    return red_patch, blue_patch, legend
     

# Return Scatter Plot with Cluster Data
def get_plot(dm0, dm1, size, dmC, plotInfo):
    
    with plt.style.context('seaborn-whitegrid'):
        
        # Plot Config
        ax, fig = plotConfig(7, 5)
       
        # Data Information
        ax.scatter(dm0, dm1, s = size, c = dmC, cmap = plt.cm.bwr)        
        ax.grid(False)
        # Annotations
        for x, y, name, gen in zip(dm0, dm1, dmC, dmC):
            ax.text(x, y, name, color=plt.cm.bwr(gen),
                     fontdict={'family': 'Arial', 'size': 15}) 
        
        # Plot Title
        plt.title(plotInfo[0])
        
        # Axis Labels
        plt.xlabel(plotInfo[1])
        plt.ylabel(plotInfo[2])
            
        # Label Patches
        plotLabels()
    
        # Plot
        plt.tight_layout()
        plt.show()
        fig.savefig('ts3t.png')