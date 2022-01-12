# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:42:19 2021

@author: lucas
"""
from ultrapyfit.utils.divers import read_data, select_traces
from ultrapyfit.experiment import Experiment
import os

path = 'C:\\Users\\lucas\\git project\\ultrafast\\examples\\data\\exp3_data_denoised.csv'

# path_2 = "C:/Users/lucas/Downloads/dataluc.csv"
path = '../data/exp3_data_denoised.csv'
path = os.path.abspath(path)

# time, data, wave = read_data(path, wave_is_row= True)

# either original_taus = [8, 30, 200]
# or original_taus = [1, 8, 30]

experiment = Experiment.load_data(path, wave_is_row=True)

# alternatively experiment can be instantiate as follow

# time, data, wave = read_data(path, wave_is_row= True)
# experiment = Experiment(time, data, wave)

# DATA EXPLORATION
# ----------------
# The first thing to do is explore the data set loaded

# To explore the spectral dimension use "plot_spectra" method
# By default all spectra will be plotted 
# (Except they are more than 250 which is the case)
experiment.plot_spectra('auto')

# From the figure we observe that 8 spectra equally spaced
# at 1520 nm are displayed.

# To plot plot a different number of traces automatically use the following format
# ['auto', 6, 1480] six spectra plotted equally spced at 1480 
experiment.plot_spectra(times=['auto', 6, 1480])

# To explore the time dimension use "plot_traces" method
# This function plots by default the selected traces except 
# If traces is set to 'auto' 10 traces separated equally in spectra dimension
# are display
experiment.plot_traces()

# alternatively we can pass a list of traces value (according to wave vector)
experiment.plot_traces([1408,1520,1578])

# To explore the number of components is possible to use the singular value decomposition (SVD)
# The SVD reveal that the data set is compose by three components
experiment.plot_full_SVD(1, True)

# After selecting the SV this can be plot again with "plot_traces" function
experiment.plot_traces()

# The original traces can be selected again with "select_traces" method
experiment.select_traces(points='all')

# To perform a global fit normally is not perform over the complete data set
# but over a selection of traces. Selecting traces is easy with "select_traces"
# function. To select a series of traces indicate the separation value.
# The selected traces will be separated 10 points in the wave vector
experiment.select_traces(points=10, average=0)

# UNITS
# -----
# Experiment automatically handles unit of figures via two attributes
# Modifying the units automatically modify figures axis and legends
# time_unit by default "ps"
# wavelength_unit by default "nm"
experiment.time_unit 
# >>> 'ps'
experiment.time_unit = 'us'
experiment.time_unit 
# >>> 'μs'
experiment.time_unit = 'millisecond'
experiment.time_unit 
# >>> 'ms'

experiment.wavelength_unit
# >>> 'nm'
experiment.wavelength_unit = 'wavenumber'
experiment.wavelength_unit
# >>> 'cm-1'
experiment.wavelength_unit = 'nanometer'
experiment.wavelength_unit
# >>> 'nm'
experiment.wavelength_unit = 'cm'
experiment.wavelength_unit
# >>> 'cm-1'

# DATA FITTING
# ------------
# The first step to perform a fitting is to initialize the model and the parameters.
# For a classical exponential fitting use "initial_exp_params" function.
# the first parameters to indicate is t0 and fwhm which indicates if the exponential
# fit is modified with a gaussian function. For this data set t0 = 0, and fwhm = None
# Since we dont have the signal raising. Then the initial guess for the fit are given.
# The SVD reveled that 3 component where needed, and values are given after plotting 
# several traces.
#                               (t0, fwhm, t1, t2, t3)
experiment.fitting.initialize_exp_params(0, None, 4, 60, 500)

# now we are ready to fit the data with "final_fit" method, this will previously
# run a prefit 
experiment.fitting.fit_global()

# RESULTS EXPLORING
# -----------------
# For any fit the first thing is to print the results which can be done with the
# "print_results" method (If none fit number is passed the last performed fit
# will be consider)
experiment.fitting.print_fit_results()

# >>> Fit number 1: 	Global Exponential fit
# >>> -------------------------------------------------
# >>> Results:	Parameter		Initial value	Final value		Vary
# >>>			time 0:   		0.0000   		0.0000   		False
# >>>			tau 1:   		4.0000   		8.0000   		True
# >>>			tau 2:   		60.0000   		30.0000   		True
# >>>			tau 3:   		500.0000   		200.0000   		True
# >>> Details:
# >>> 		Number of traces: 30; average: 0
# >>> 		Fit with 3 exponential; Deconvolution False
# >>> 		Tau inf: None
# >>> 		Number of iterations: 994
# >>> 		Number of parameters optimized: 240
# >>> 		Weights: False

# The following step is plot the fit and residual plot which is done with the 
# "plot_fit" method
experiment.fitting.plot_global_fit()

# finally the Decay Associated Spectra (DAS) can be plotted with the plot_DAS method
experiment.fitting.plot_DAS()

