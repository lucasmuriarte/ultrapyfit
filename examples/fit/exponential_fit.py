# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrafast.fit.ExponentialFit import globalfit_exponential
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.fit.GlobalFitBootstrap import BootStrap
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.fit.GlobalParams import GlobExpParameters

path = '../data/denoised_2.csv'
time, data, wave = read_data(path, wave_is_row= True)

# original_taus = [8, 30, 200]
data_select, wave_select = select_traces(data, wave, 'auto')
result = globalfit_exponential(time, data_select, 4, 40, 400, t0 =2)


explorer = ExploreResults(result)
explorer.print_results()
explorer.plot_fit()
explorer.plot_DAS()




