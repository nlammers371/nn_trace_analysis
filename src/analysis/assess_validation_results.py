from bokeh.charts import Scatter, output_file, show, Area
from bokeh.plotting import figure, show, output_file, gridplot
from bokeh.models import Range1d

#from bokeh.palettes import Category20b as palette #@UnresolvedImport

import pandas as pd
import os
from bokeh.palettes import Spectral11
import numpy as np
import itertools
from itertools import chain
import sys
from Bio import SeqIO
from Bio.Seq import Seq
import time

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import matplotlib.gridspec as gridspec # subplot


datapath = "../output/train/train_016_2017-04-30_02:08:02/evaluate/"
outpath = os.path.join(datapath,"plots")
fname = "eve2_3state_ap41_slow"
TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"
outname = fname + "_plots"
pd_data = pd.read_csv(os.path.join(datapath,fname + ".csv"))
pd_data.columns = ["ID", "Fluo", "label", "prediction_max","prediction_avg"]

plot_list = list(set(pd_data.ID.values))

mycolors=Spectral11[0:10]
if not os.path.isdir(outpath):
    os.makedirs(outpath)

for i in xrange(10):
    plot_data = pd_data[pd_data.ID==plot_list[i]]
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    x_axis = range(len(plot_data.index))
    ax.plot(x_axis,plot_data["label"],lw=2,color=mycolors[0])
    #ax.plot(x_axis, plot_data["prediction_max"],lw=2,linestyle='--', color=mycolors[2])
    ax.plot(x_axis, plot_data["prediction_avg"],lw=3,linestyle='-.',color=mycolors[3])

    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Course of Training Period')

    fig.savefig(os.path.join(outpath,outname + "_" + str(i) + ".png"))
    plt.close()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    x_axis = range(len(plot_data.index))
    ax.plot(x_axis, plot_data["label"], lw=2, color=mycolors[0])
    ax.plot(x_axis, plot_data["prediction_max"], lw=2, linestyle='--', color=mycolors[2])
    #ax.plot(x_axis, plot_data["prediction_avg"], lw=3, linestyle='-.', color=mycolors[5])
    ax.plot(x_axis, plot_data["Fluo"], lw=3, linestyle='-.', color=mycolors[3])

    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Course of Training Period')

    fig.savefig(os.path.join(outpath, outname + "_full_" + str(i) + ".png"))
    plt.close()

    #plt.show()