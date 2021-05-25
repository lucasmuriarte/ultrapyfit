# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:17:57 2021

@author: lucas
"""
path_2 = "C:/Users/lucas/Downloads/dataluc.csv"
irf_path = "C:/Users/lucas/Downloads/IRF_330nm_36ps_20000.dat"

import pandas as pd
import numpy as np
from ultrafast.fit.GlobalFit import GlobalFitWithIRF, GlobalFitExponential
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.fit.GlobalParams import GlobExpParameters
from ultrafast.fit.ModelCreator import ModelCreator
import matplotlib.pyplot as plt

data = pd.read_csv(path_2, header=None).values
irf = pd.read_csv(irf_path, header=9).values[:1500,0]
time = np.linspace(0,1500,1500)*0.004
params = GlobExpParameters(data.shape[1], [0.6])
params.adjustParams(0, False, None)
parameters = params.params

fitter = GlobalFitWithIRF(time, data, irf, 1, parameters, wavelength=None)
fitter.allow_stop = True
result = fitter.global_fit(maxfev=5000)

explorer = ExploreResults(result)
fig, ax = explorer.plot_fit()
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_ylim(0, 25000)
model = ModelCreator(1, time)
plt.show()

parameters['pre_exp1_1'].value=10
y = model.expNDatasetIRF(parameters, 0, irf)


plt.figure()        
#def jumpexpmodel(x,tau1,ampl1,y0,x0,args=(irf))
plt.semilogy(time,y,'r--',time,data[:,1],'bo')
plt.title("test the model")


fitter = GlobalFitExponential(time, data, 1, parameters, False, wavelength=None)
fitter.allow_stop = True
result = fitter.global_fit(maxfev=5000)

