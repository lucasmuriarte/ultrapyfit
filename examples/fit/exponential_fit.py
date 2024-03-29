# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrapyfit.old.ExponentialFit import globalfit_exponential
from ultrapyfit.utils.divers import read_data, select_traces
from ultrapyfit.graphics.ExploreResults import ExploreResults

path = '../data/denoised_2.csv'
time, data, wave = read_data(path, wave_is_row= True)

# original_taus = [8, 30, 200]
data_select, wave_select = select_traces(data, wave, 'auto')
result = globalfit_exponential(time, data_select, 4, 40, 400, t0 =2)


explorer = ExploreResults(result)
explorer.print_fit_results()
explorer.plot_global_fit()
explorer.plot_DAS()




