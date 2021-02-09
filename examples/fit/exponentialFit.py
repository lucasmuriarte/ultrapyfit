# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrafast.GlobalFitClass import globalfit_exponential
from ultrafast.outils import readData, select_traces

path = 'examples/denoised_2.csv'
time, data, wave = readData(path, wave_is_row= True)

original_taus = [8, 30, 200]
data_select, wave_select = select_traces(data, wave, 'auto')
result = globalfit_exponential(time, data_select, 4, 40, 400)
params_result = result.params

final_taus = [params_result['tau1_1'], params_result['tau2_1'], params_result['tau3_1']]

