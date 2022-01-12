# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:18:19 2021

@author: lucas
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrapyfit.fit.GlobalFit import GlobalFitExponential
from ultrapyfit.experiment import Experiment
from ultrapyfit.utils.divers import read_data
import matplotlib.pyplot as plt

path='D:/PC/donnes/donnes femto/DATA RAL MARS 2019 oxford/preproces december2019 with bining/Off to On TRIMPS correct/WT.csv'
experiment = Experiment.load_data(path, wave_is_row=False)
experiment.calibrate_wavelength([1494.0, 1518.0, 1633.0, 1687.0],
                                [1491.0, 1515.0, 1633.0, 1681.0])

experiment.cut_wavelength(1450, 1720, 'select')
experiment.cut_time(0.001,2)
experiment.time_unit = 'ns'
experiment.wavelength_unit = 'cm'
experiment.plot_spectra('auto')
experiment.preprocessing_report.print()
experiment.derivate_data(5)
experiment.initialize_exp_params(0.001, None, 0.001, 0.004, 0.08)
experiment.fit_global([False, True, True], )
# res.save('./fit')
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.restore_data('cut_time')
experiment.cut_time(1)
experiment.derivate_data(5)
experiment.initialize_exp_params(2, None, 100, 3000, 80000)
experiment.fit_global([True, True, True], )
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.initialize_exp_params(2, None, 100, 3000, 80000, 2050000, y0=0)
experiment.fit_global([True, True, True, False], )
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.plot_global_fit()
experiment.initialize_exp_params(2, None, 100, 3000, 80000, 2050000, y0=0)
experiment.fit_global([True, True, True, True], )
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.plot_global_fit()
# # original_taus = [8, 30, 200]
# params = GlobExpParameters(data.shape[1], [4, 40, 400])
# params.adjustParams(0.1, False, None)
# parameters = params.params

path2 = 'C:/Users/lucas/Downloads/calibrated spectra.asc'
experiment2 = Experiment.load_data(path2)
experiment2.chirp_correction_graphically('polynomial')



#leucine
path='D:/PC/donnes/donnes femto/DATA RAL MARS 2019 oxford/preproces december2019 with bining/Off to On TRIMPS correct/leucine.csv'
experiment = Experiment.load_data(path, wave_is_row=False)
experiment.calibrate_wavelength([1494.0, 1518.0, 1633.0, 1687.0],
                                [1491.0, 1515.0, 1633.0, 1681.0])


# experiment.subtract_polynomial_baseline([1296.31,1402.98,1455.38,1600.81,1677.85,1717.37,1770.45],2)
experiment.cut_wavelength(1450, 1720, 'select')
experiment.cut_time(2)                               
experiment.derivate_data(5)
experiment.time_unit = 'ns'
experiment.wavelength_unit = 'cm'
experiment.initialize_exp_params(2, None, 100, 3000, 20000)
experiment.fit_global([True, True, True])
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.plot_global_fit()
plt.xscale('log')

#alanine
path='D:/PC/donnes/donnes femto/DATA RAL MARS 2019 oxford/preproces december2019 with bining/Off to On TRIMPS correct/alanine.csv'
experiment = Experiment.load_data(path, wave_is_row=False)
experiment.calibrate_wavelength([1494.0, 1518.0, 1633.0, 1687.0],
                                [1491.0, 1515.0, 1633.0, 1681.0])


# experiment.subtract_polynomial_baseline([1296.31,1402.98,1455.38,1600.81,1677.85,1717.37,1770.45],2)
experiment.cut_wavelength(1450, 1720, 'select')
experiment.cut_time(2)                               
experiment.derivate_data(5)
experiment.time_unit = 'ns'
experiment.wavelength_unit = 'cm'
experiment.initialize_exp_params(2, None, 100, 3000)
experiment.fit_global([True, True])
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.plot_global_fit()
plt.xscale('log')