# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrafast.fit.GlobalFit import GlobalFitExponential
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.fit.GlobalParams import GlobExpParameters

path = 'denoised_2.csv'
time, data, wave = read_data(path, wave_is_row= True)

# original_taus = [8, 30, 200]
data_select, wave_select = select_traces(data, wave, 5)
params = GlobExpParameters(data_select.shape[1], [4, 40, 400])
params.adjustParams(0, False, None)
parameters = params.params

fitter = GlobalFitExponential(time, data_select, 3, parameters, False,
                              wavelength=wave_select)
fitter.allow_stop = True
result = fitter.global_fit()

explorer = ExploreResults(result)
explorer.print_results()
explorer.plot_fit()
explorer.plot_DAS()




