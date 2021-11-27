# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:45:45 2021

@author: lucas
"""
from ultrafast.utils.ChirpCorrection import EstimationGVDPolynom, EstimationGVDSellmeier
from ultrafast.utils.divers import read_data, select_traces
import matplotlib.pyplot as plt

#TODO Change path here
# path = 'C:/Users/lucas/Downloads/calibrated spectra.asc'
path = 'examples/data/data.asc'
time, data, wave = read_data(path)


corrector = EstimationGVDSellmeier(time, data, wave, 400)
corrector.estimate_GVD_from_grath()

plt.show()
