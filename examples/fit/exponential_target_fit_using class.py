# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrafast.fit.GlobalFit import GlobalFitTarget
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.fit.GlobalParams import GlobalTargetParameters
from ultrafast.graphics.targetmodel import Model


path = 'C:/Users/lucas/git project/ultrafast/examples/data/data for igor 3 exp.txt'
time, data, wave = read_data(path, wave_is_row=True, separator='\t')

# original_taus = [8, 30, 200]
model = Model.load("C:/Users/lucas/git project/ultrafast/tests/ultrafast/fit/testmodel2.model")
params_model = model.genParameters()
exp_no = params_model['exp_no']
data_select, wave_select = select_traces(data, wave, 'auto')
params = GlobalTargetParameters(data_select.shape[1], model)
params.adjustParams(0, False, None)
parameters = params.params

fitter = GlobalFitTarget(time, data_select, 3, parameters, False,
                              wavelength=wave_select)
result = fitter.global_fit()

explorer = ExploreResults(result)
explorer.print_results()
explorer.plot_fit()
explorer.plot_DAS()




