# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:55:03 2020

@author: lucas
"""
import numpy as np
from ultrafast.ExploreResultsClass import ExploreResults
from ultrafast.PlotTransientClass import ExploreData
from ultrafast.GlobExpParams import GlobExpParameters
from ultrafast.outils import define_weights, UnvariableContainer, LabBook, book_annotate
from ultrafast.PreprocessingClass import Preprocessing
from ultrafast.GlobalFitClass import GlobalFitExponential
from ultrafast.GlobalTargetClass import GlobalFitTargetModel
import os


class ExperimentException(Exception):
    """General Purpose Exception."""
    
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return "{}".format(self.msg)


class Experiment(ExploreData, ExploreResults):
    def __init__(self, x, data, wavelength=None):
        if wavelength is None:
            self.wavelength = wavelength
            general_cal = '\tNone'
        else:
            self.wavelength = np.array(wavelength)
            general_cal = '\tAlready calibrated'
        self.x = x
        self.data = data
        self.selected_traces = data
        self.selected_wavelength = wavelength
        self.data_sets = UnvariableContainer()
        self.excitation = None
        self.tau_inf = 1E12
        self.GVD_corrected = False
        self.action_records = UnvariableContainer(name="Sequence of actions")
        self.fit_records = UnvariableContainer(name="Fits")
        self.params = None
        self.weights = {'apply': False, 'vector': None, 'range': [], 'type': 'constant', 'value': 2}
        self.cmap = 'viridis'
        self.preprocessing_report = LabBook(name="Pre-processing")
        self._averige_selected_traces = 0
        self._deconv = True
        self._exp_no = 1
        self._fit_number = 0
        self._params_initialized = False
        self._last_data_sets = None
        self._initialized()
        ExploreData.__init__(self,self.x,self.data,self.wavelength,self.selected_traces,self.selected_wavelength,self.cmap)
        ExploreResults.__init__(self,self.fit_records.global_fits)
    
    @staticmethod
    def load():
        pass
    
    def _initialized(self):
        self.data_sets.original_data = UnvariableContainer(time=self.x, data=self.data, wavelength=self.wavelength)
        self.fit_records.single_fits = {}
        self.fit_records.bootstrap_record = {}
        self.fit_records.conf_interval = {}
        self.fit_records.target_models = {}
        self.fit_records.global_fits = {}
        self.baseline_substraction = book_annotate(self.preprocessing_report)(self.baseline_substraction)
        self.cut_time = book_annotate(self.preprocessing_report)(self.cut_time)
        self.average_time = book_annotate(self.preprocessing_report)(self.average_time)
        self.derivate_data = book_annotate(self.preprocessing_report)(self.derivate_data)
        self.cut_wavelength = book_annotate(self.preprocessing_report)(self.cut_wavelength)
        self.del_points = book_annotate(self.preprocessing_report)(self.del_points)
        self.shitTime = book_annotate(self.preprocessing_report)(self.shit_time)
    
    @property
    def chirp_corrected(self):
        return self.GVD_corrected
    
    @chirp_corrected.setter
    def chirp_corrected(self, value):
        if type(value) == bool:
            self.GVD_corrected = value

    @property
    def type_fit(self):
        if self._params_initialized == False:
            return "Not ready to fit data"
        else:
            return f"parameters for {self._params_initialized} fit  with {self._exp_no} components"

    @type_fit.setter
    def type_fit(self, value):
        pass

    def _add_to_data_set(self, key):
        if hasattr(self.data_sets, key):
            pass
        else:
            container = UnvariableContainer(time=self.x, data=self.data, wavelength=self.wavelength)
            self.data_sets.__setattr__(key, container)
            self._last_data_sets = container
        
    def _add_action(self, value):
        val = len(self.action_records.__dict__)
        self.action_records.__setattr__(f"_{val+1}", value)
        
    def baseline_substraction(self, number_spec=2, only_one=False):
        self._add_to_data_set("before_baseline_substraction")
        self._add_action("baseline substraction")
        new_data = Preprocessing.baseline_substraction(self.data, number_spec=number_spec, only_one=only_one)
        self.data = new_data

    def cut_time(self, mini=None,maxi=None):
        self._add_to_data_set("before_cut_time")
        self._add_action("cut time")
        new_data, new_x = Preprocessing.cut_rows(self.data, self.x, mini, maxi)
        self.data, self.x = new_data, new_x

    def average_time(self, starting_point, step, method='log', grid_dense=5):
        self._add_to_data_set("before_average_time")
        self._add_action("average time")
        new_data, new_x = Preprocessing.average_time_points(self.data, self.x, starting_point, step, method, grid_dense)
        self.data, self.x = new_data, new_x

    def derivate_data(self, window_length=25, polyorder=3, deriv=1, mode='mirror'):
        self._add_to_data_set("before_derivation")
        self._add_action("derivate data")
        new_data = Preprocessing.derivateData(window_length, polyorder, deriv, mode)
        self.data = new_data

    def cut_wavelength(self, left=None, right=None, innercut=False):
        self._add_to_data_set("before_cut_wavelength")
        self._add_action("cut wavelength")
        new_data, new_wave = Preprocessing.cut_columns(self.data, self.wavelength, left, right, innercut)
        self.data, self.wavelength = new_data, new_wave

    def del_points(self, points, dimension='time'):
        self._add_to_data_set("before_delete_point")
        self._add_action(f"delete point {dimension}")
        if dimension == 'time':
            new_data, new_x = Preprocessing.del_points(points, self.data, self.x)
        elif dimension == 'wavelength':
            new_data, new_wave = Preprocessing.del_points(points, self.data, self.wavelength)
        else:
            raise ExperimentException('dimension should be "time" or "wavelength"')
        self.data, self.x = new_data, new_x

    def shit_time(self, value):
        self._add_to_data_set("before_time_shift")
        self._add_action("shift time")
        self.x = self.x-value
    
    def define_weights(self, rango, typo='constant', val=5):
        self._add_action("define weights")
        self.weights = define_weights(self.x, rango, typo=typo, val=val)

    def createNewDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.working_directory = path
        self.save['path']=self.working_directory
    
    def select_traces(self, points=10, average=1, avoid_regions=None):
        super().select_traces(points, average, avoid_regions)
        self._readapt_params()
        self._averige_selected_traces = average if points != 'all' else 0
        
    def _readapt_params(self):
        if self._params_initialized == 'Exponential':
            previous_taus = self._last_params['taus']
            t0 = self._last_params['t0']
            fwhm = self._last_params['fwhm']
            tau_inf = self._last_params['tau_inf']
            opt_fwhm = self._last_params['opt_fwhm']
            self.initialize_exp_params(t0, fwhm, *previous_taus, tau_inf=tau_inf, opt_fwhm=opt_fwhm)
        elif self._params_initialized == 'Target':
            print('to be coded')
            # to do
            # self.initialize_target_params()
        else:
            pass
    
    def initialize_exp_params(self, t0, fwhm, *taus, tau_inf=1E12, opt_fwhm=False):
        taus = list(taus)
        self._last_params = {'t0': t0, 'fwhm': fwhm, 'taus': taus, 'tau_inf': tau_inf, 'opt_fwhm': opt_fwhm}
        self._exp_no = len(taus)
        param_creator = GlobExpParameters(self.selected_traces.shape[1], taus)
        if fwhm is None:
            self._deconv = False
            vary_t0 = False
        else:
            vary_t0 = True
            self._deconv = True
            self.tau_inf = tau_inf
        param_creator.adjustParams(t0, vary_t0, fwhm, opt_fwhm, self.GVD_corrected, tau_inf)
        self.params = param_creator.params
        self._params_initialized = 'Exponential'
        self._add_action(f'new {self._params_initialized} parameters initialized')
        
    def initialize_target_params(self, t0, fwhm, *taus, tau_inf=1E12, opt_fwhm=False):
            pass
        
    def final_fit(self, vary=True, maxfev=5000, apply_weights=False):
        if self._params_initialized == 'Exponential':
            minimizer = GlobalFitExponential(self.x, self.selected_traces, self._exp_no, self.params,
                                             self._deconv, self.tau_inf, GVD_corrected=self.GVD_corrected)
        elif self._params_initialized == 'Target':
            minimizer = GlobalFitTargetModel(self.selected_traces, self.selected_wavelength, self.params)
        else:
             raise ExperimentException('Parameters need to be initiliazed first"')
        if apply_weights:
            minimizer.weights = self.weights
        self._fit_number += 1
        results = minimizer.finalFit(vary, maxfev, apply_weights)
        results.details['svd_fit'] = self._SVD_fit
        results.wavelength = self.selected_wavelength
        results.details['avg_traces'] =  self._averige_selected_traces
        self.fit_records.global_fits[self._fit_number] = results
        self._add_action(f'{self._params_initialized} fit performed')
        
        
        
        
        
        
        