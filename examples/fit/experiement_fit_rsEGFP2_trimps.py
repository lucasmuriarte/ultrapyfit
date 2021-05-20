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
from ultrafast.fit.GlobalFit import GlobalFitExponential
from ultrafast.experiment import Experiment
from ultrafast.utils.divers import read_data

path='E:/PC/donnes/donnes femto/DATA RAL MARS 2019 oxford/preproces december2019 with bining/Off to On TRIMPS correct/WT.csv'
experiment = Experiment.load_data(path, wave_is_row=False)
experiment.calibrate_wavelength([1494.0, 1518.0, 1633.0, 1687.0],
                                [1491.0, 1515.0, 1633.0, 1681.0])
experiment.cut_wavelength(1440, 1720, 'select')
experiment.cut_time(0.001,2)
experiment.time_unit = 'ns'
experiment.wavelength_unit = 'cm'
experiment.plot_spectra('auto')
experiment.preprocessing_report.print()
experiment.derivate_data(5)
experiment.initialize_exp_params(0.001, None, 0.001, 0.004, 0.08)
experiment.global_fit([False, True, True],)
# res.save('./fit')
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.restore_data('cut_time')
experiment.cut_time(1)
experiment.derivate_data(5)
experiment.initialize_exp_params(2, None, 100, 3000, 80000)
experiment.global_fit([True, True,True],)
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.initialize_exp_params(2, None, 100, 3000, 80000, 2050000, y0=0)
experiment.global_fit([True, True,True, False],)
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.plot_fit()
experiment.initialize_exp_params(2, None, 100, 3000, 80000, 2050000, y0=0)
experiment.global_fit([True, True,True, True],)
experiment.plot_DAS(plot_integrated_DAS=True)
experiment.plot_fit()
# # original_taus = [8, 30, 200]
# params = GlobExpParameters(data.shape[1], [4, 40, 400])
# params.adjustParams(0.1, False, None)
# parameters = params.params



