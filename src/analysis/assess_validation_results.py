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


datapath = "../output/train/train_015_2017-04-29_06:15:51/evaluate/eve2_3state_ap41.csv"
TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"

pd_data = pd.read_csv(datapath)
pd_data.columns = ["ID", "Fluo", "label", "prediction","blah"]

plot_list = list(set(pd_data.ID.values))

mypalette=Spectral11[0:2]

for i in xrange(10):
    plot_data = pd_data[pd_data.ID==plot_list[i]]
    print(pd.DataFrame.head(plot_data))
    p_x = range(500)
    p_y_list = ["label", "prediction"]

    p = figure(tools=TOOLS, title="Predicted Fluo vs. Actual", height = 600, width=1200)
    p.multi_line(xs=[p_x]*2, ys=[plot_data[j].values for j in p_y_list],  line_color=mypalette)

    show(p)