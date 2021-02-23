# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrafast.fit.ExponentialFit import globalfit_exponential
from ultrafast.utils.divers import read_data, select_traces

path = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_denoised_2.csv'
time, data, wave = read_data(path, wave_is_row= True)

original_taus = [8, 30, 200]
data_select, wave_select = select_traces(data, wave, 'auto')
result = globalfit_exponential(time, data_select, 4, 40, 400)
params_result = result.params

final_taus = [params_result['tau1_1'], params_result['tau2_1'],
              params_result['tau3_1']]

