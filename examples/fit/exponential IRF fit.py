# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:17:57 2021

@author: lucas
"""
path_2 = "C:/Users/lucas/Downloads/dataluc.csv"
irf_path = "C:/Users/lucas/Downloads/IRF_330nm_36ps_20000.dat"

import pandas as pd
import numpy as np
from ultrapyfit.fit.GlobalFit import GlobalFitWithIRF, GlobalFitExponential
from ultrapyfit.graphics.ExploreResults import ExploreResults
from ultrapyfit.fit.GlobalParams import GlobExpParameters
from ultrapyfit.fit.ModelCreator import ModelCreator
import matplotlib.pyplot as plt

data = pd.read_csv(path_2, header=None).values
irf = pd.read_csv(irf_path, header=9).values[:1500,0]
time = np.linspace(0,1500,1500)*0.004
params = GlobExpParameters(data.shape[1], [0.6])
params.adjustParams(10, False, None)
parameters = params.params
parameters['t0_1'].vary=True
for i in range(8):
    parameters['pre_exp1_%i' %(i+1)].value=500
    if i > 0:
        parameters['t0_%i' %(i+1)].expr=None
        parameters['t0_%i' %(i+1)].value=10
        parameters['t0_%i' %(i+1)].vary=True


fitter = GlobalFitWithIRF(time, data, irf, 1, parameters, wavelength=None)
# fitter.allow_stop = True
result = fitter.global_fit(maxfev=5000)

explorer = ExploreResults(result)
fig, ax = explorer.plot_global_fit()
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_ylim(1, 25000)
plt.show()


model = ModelCreator(1, time)

def initial_model(curve, params, tau=570, pre_exp=10, t0=0):
    #params.pretty_print()
    params['tau1_1'].value=tau
    params['t0_1'].value=t0
    params['pre_exp1_%i' %curve].value=pre_exp
    y = model.expNDatasetIRF(params, curve-1, irf)


    plt.figure()        
    #def jumpexpmodel(x,tau1,ampl1,y0,x0,args=(irf))
    plt.semilogy(time,y,'r--',time,data[:,curve+1],'bo')
    plt.title("test the model")
    plt.show()

initial_model(2,parameters, 0.5, 5, 50)

params = GlobExpParameters(data.shape[1], [0.6])
params.adjustParams(0.3, False, None)
parameters = params.params
fitter = GlobalFitExponential(time, data, 1, parameters, False, wavelength=None)
fitter.allow_stop = True
result = fitter.global_fit(maxfev=5000)

