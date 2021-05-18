# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:55:03 2020

@author: lucas
"""
import numpy as np
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.graphics.ExploreData import ExploreData
from ultrafast.fit.GlobalParams import GlobExpParameters
from ultrafast.utils.ChirpCorrection_redone import EstimationGVDPolynom, EstimationGVDSellmeier
from ultrafast.utils.divers import define_weights, UnvariableContainer, LabBook,\
    book_annotate, read_data, TimeUnitFormater, select_traces
from ultrafast.utils.Preprocessing import Preprocessing, ExperimentException
from ultrafast.fit.GlobalFit import GlobalFitExponential, GlobalFitTarget
import os
from matplotlib.offsetbox import AnchoredText
import copy
import pickle

def capture_chirp_correction(obj):
    print(obj._params_initialized)
    pass

class Experiment(ExploreData, ExploreResults):
    """
    Class to work with a time resolved data set and easily preprocess the data, obtain
    quality fits and result, explore the data set and keep track of actions done. Finally
    to easily create figures already formatted.
    The class inherits ExploreData and ExploreResults therefore all methods for plotting and
    exploring a data set, explore the SVD space and check results are available.

    Attributes
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2darray
        Array containing the data, the number of rows should be equal to the len(x) and the
        number of columns equal to len(wavelength)

    wavelength: 1darray
        Wavelength vector

    selected_traces: 2darray
        Sub dataset of data. The global fits are performed only in the selected traces data set.
        Preprocesing actions are done on both data and selected traces. To select a part of the
        data set, use select_traces, select_traces_graph. To select the SVD left values for
        fitting use select_SVD_vectors or plot_SVD(1, select=True).

    selected_wavelength: 1darray
        Sub dataset of wavelength. Automatically updated when selected traces is update

    time_unit: str (default ps)
        Contains the strings of the time units to format axis labels and legends automatically:
        time_unit str of the unit. >>> e.g.: 'ps' for picosecond
        Can be passes as kwargs when instantiating the object

    wavelength_unit: str (default nm)
        Contains the strings of the wavelength units to format axis labels and legends automatically:
        wavelength_unit str of the unit: >>> e.g.: 'nm' for nanometers
        Can be passes as kwargs when instantiating the object

    params: lmfit parameters object
        object containing the initial parameters values used to build an exponential model.
        These parameters are iteratively optimize to reduce the residual matrix formed by
        data-model (error matrix) using Levenberg-Marquardt algorithm. After a global fit
        they are actualized to the results obtained. After selection of new traces, this are
        automatically re-adapted.

    weights: dict
        contains the weigthing vector that will be apply if apply_weigths is set
        to True in any of the fitting functions. The weight can be define with
        the define weights function.

    GVD_corrected/chirp_corrected: (default False)
        Indicates if the data has been chirp corrected. It set to True after
        calling the chrip_correction method.

    preprocessing_report: class LabBook
        Object containing and keeping track of the preprocessing actions done to the data set.
        The preprocessing_report.print() method will print the status of the actions done and
        parameters passed to each of the preprocesing functions. The general report method of the
        Experiment class will also print it.

    action_records: class UnvariableContainer
        Object containing and keeping track of the important actions done. similar to the
        preprocessing_report t content can be printed. The general report method of the
        Experiment class will also print it.

    fit_records: class UnvariableContainer
        Object containing and keeping track of fit results done. The attributes are: single_fits,
        global_fits, integral_band_fits, bootstrap_record, conf_interval and target_models

    excitation: float or int (default None)
        If given the spectra figures will display a white box ± 10 units of excitation automatically
        to cover the excitation if this value is between the wavelength range. This value is needed
        for some chirp correction methods. If the excitation is not needed.
    """
    def __init__(self, x, data, wavelength=None, path=None, **kwargs):
        """
        Constructor function to initialize the Experiment class


        Parameters
        ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            array containing the data, the number of rows should be equal to the len(x)

        wavelength: 1darray (default None)
            Wavelength vector. Alternatively the wavelength can be "None" and may be later on calibrated.
            If None, wavelength vector will be an array from 0 to the length of data columns. Although
            it can be initialize without wavelength vector, is recommended to pass this value or calibrate
            in a first step.

        path: str (default None)
            String to keep track of the data loaded for keeping track in future.

        """
        units = dict({'time_unit': 'ps', 'wavelength_unit': 'nm'}, **kwargs)
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
        self.GVD_corrected = False
        self.action_records = UnvariableContainer(name="Sequence of actions")
        self.fit_records = UnvariableContainer(name="Fits")
        self.params = None
        self.weights = {'apply': False, 'vector': None, 'range': [], 'type': 'constant', 'value': 2}
        self.preprocessing_report = LabBook(name="Pre-processing")
        self.data_path = path
        self._units = units
        self._averige_selected_traces = 0
        self._deconv = True
        self._exp_no = 1
        self._fit_number = 0
        self._params_initialized = False
        self._last_data_sets = None
        self._tau_inf = 1E12
        self._initialized()
        self._chirp_corrector = None
        ExploreData.__init__(self, self.x, self.data, self.wavelength, self.selected_traces, self.selected_wavelength,
                             'viridis', **self._units)
        ExploreResults.__init__(self, self.fit_records.global_fits, **self._units)

    def _initialized(self):
        """
        Finalize the initialization of the Experiment class
        """
        self.data_sets.original_data = UnvariableContainer(time=self.x, data=self.data, wavelength=self.wavelength)
        self.preprocessing_report._last_action = None
        self.fit_records.single_fits = {}
        self.fit_records.bootstrap_record = {}
        self.fit_records.conf_interval = {}
        self.fit_records.target_models = {}
        self.fit_records.global_fits = {}
        self.fit_records.integral_band_fits = {}
        self.baseline_substraction = book_annotate(self.preprocessing_report)(self.baseline_substraction)
        # self.chirp_correction = book_annotate(self.preprocessing_report)(self.chirp_correction)
        self.subtract_polynomial_baseline = book_annotate(self.preprocessing_report)(self.subtract_polynomial_baseline)
        self.cut_time = book_annotate(self.preprocessing_report)(self.cut_time)
        self.average_time = book_annotate(self.preprocessing_report)(self.average_time)
        self.derivate_data = book_annotate(self.preprocessing_report)(self.derivate_data)
        self.cut_wavelength = book_annotate(self.preprocessing_report)(self.cut_wavelength)
        self.del_points = book_annotate(self.preprocessing_report)(self.del_points)
        self.shitTime = book_annotate(self.preprocessing_report)(self.shit_time)
        self._unit_formater = TimeUnitFormater(self._units['time_unit'])

    @property
    def chirp_corrected(self):
        return self.GVD_corrected

    @chirp_corrected.setter
    def chirp_corrected(self, value):
        if type(value) == bool:
            self.GVD_corrected = value

    @property
    def type_fit(self):
        if not self._params_initialized:
            return "Not ready to fit data"
        else:
            return f"parameters for {self._params_initialized} fit  with {self._exp_no} components"

    @type_fit.setter
    def type_fit(self, value):
        pass

    @staticmethod
    def load_data(path: str, wavelength=0, time=0, wave_is_row=False, separator=',', decimal='.'):
        x, data, wave = read_data(path, wavelength, time, wave_is_row, separator, decimal)
        return Experiment(x, data, wave, path)

    @staticmethod
    def load(path: str):
        """
        Load a saved experiment
        """
        with open(path) as file:
            experiment = pickle.load(file)
        if type(experiment) == Experiment:
            return experiment
        else:
            msg = 'File do not correspond to a ultrafast Experiment class'
            raise ExperimentException(msg)

    def save(self, path):
        """
        Saved the current experiment
        """
        ## ToDo
        pass

    def describe_data(self):
        """
        Print description of the data
        """
        name = 'data description'
        print(f'\t {name}')
        print('-'*(len(name)+10))
        print('\tData set\t\tNº traces\tNº spectra')
        name = ['Loaded data','Current data']
        for ii, i in enumerate([self.data_sets.original_data.data, self.data]):
            print(f'\t{name[ii]}:\t {i.shape[1]}\t\t\t{i.shape[0]}')
        print(f'\tSelected traces: {self.selected_traces.shape[1]}')
        print('\n\t units')
        print('-' * (len('units') + 10))
        print(f'\tTime unit: {self.time_unit}')
        print(f'\tWavelength unit: {self.wavelength_unit}')

    def print_results(self, fit_number=None):
        """
        Print out a summarize result of a global fit.

        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None
            the last fit in  will be considered.
        """
        if fit_number is None:
            fit_number = max(self._fits.keys())
        super().print_results(fit_number=fit_number)
        if fit_number in self.fit_records.bootstrap_record.keys():
            print('\t The error has been calculated by bootstrap')
        if fit_number in self.fit_records.bootstrap_record.keys():
            print('\t The error has been calculated by an F-test')
        print('\n')

    def general_report(self, output_file=None):
        """
        Print the general report of the experiment 
        """
        self.describe_data()
        print('============================================\n')
        self.preprocessing_report.print()
        print('============================================\n')
        for i in range(len(self.fit_records.global_fits)):
            self.print_results(i+1)
        print('============================================\n')
        self.action_records.print(False, True, True)

    def calibrate_wavelength(self):
        ## TODO
        pass

    def chirp_correction_graphically(self, method, excitation=None):
        if method == 'sellmeier':
            if excitation is None:
                msg = 'The excitation must be defined'
                raise ExperimentException(msg)
            self._chirp_corrector = EstimationGVDSellmeier(self.x, 
                                                           self.data, 
                                                           self.wavelength, 
                                                           excitation)
        elif method == 'polynomila':
            self._chirp_corrector = EstimationGVDPolynom(self.x, 
                                                         self.data, 
                                                         self.wavelength)
        else:
            msg = 'Method can only be "sellmeier" or "polynomial"'
            raise ExperimentException(msg)
        self._chirp_corrector.estimate_GVD_from_grath()
        self.data = self._chirp_corrector.corrected_data
        # capture_chirp_correction(self)

    def GVD_correction_graphically(self, method):
        """
        Identical to chirp_correction method
        """
        self.chirp_correction_graphically(method)

    def baseline_substraction(self, number_spec=2, only_one=False):
        """
        Subtract a initial spectrum or an average of initial spectra
        (normally before time = 0) to the entire data set.


        Parameters
        ----------
        number_spec: int or list
            This parameter defines the number of spectra to be subtracted.
            If int should be the index of the spectrum to be subtracted
            If list should be of length 2 and contain initial and final spectra
            index that want to be subtracted

        only_one: bool (default False)
            if True only the spectrum (at index number_spec) is subtracted
            if False an average from the spectrum 0 to number_spec is subtracted
            (only applicable if number_spec is an int)

        e.g.1: number_spec = [2,5] an average from spectra 2 to 5 is subtracted
                (where 2 and 5 are included)

        e.g.2: number_spec = 5 an average from spectra 0 to 5 is subtracted
                if only_one = False; if not only spectrum 5 is subtracted
        """
        self._add_to_data_set("before_baseline_substraction")
        new_data = Preprocessing.baseline_substraction(self.data, number_spec=number_spec, only_one=only_one)

        self.data = new_data
        self._add_action("baseline substraction", True)

    def subtract_polynomial_baseline(self, points, order=3):
        """
        Fit and subtract a polynomial to the data in the spectral range (rows).
        This function can be used to correct for baseline fluctuations typically
        found in time resolved IR spectroscopy.


        Parameters
        ----------
        points: list
            list containing the wavelength values where the different transient
            spectra should be zero

        order: int or float (default: 3)
           order of the polynomial fit
        """
        self._add_to_data_set("before_subtract_polynomial_baseline")
        new_data = Preprocessing.subtract_polynomial_baseline(self.data, self.wavelenght,
                                                              points=points, order=order)
        self.data = new_data
        self._add_action("Subtracted polynomial baseline", True)

    def cut_time(self, mini=None, maxi=None):
        """
        Cut time point of the data set according to the closest values of mini
        and maxi margins given to the time vector. Contrary to cut_wavelength
        function, inner cut are not available since in time resolved
        spectroscopy is not logical to cut a complete area of recorded times.
        Therefore, giving mini and maxi margins will result in selection of
        inner time values.
        (The function assumes rows vector is sorted from low to high values)


        Parameters
        ----------
        mini: int, float or None (default: None)
          data higher than this value is kept

        maxi: int, float or None (default: None)
          data lower than this value is kept
        """
        self._add_to_data_set("before_cut_time")
        new_data, new_x = Preprocessing.cut_rows(self.data, self.x, mini, maxi)
        self.data, self.x = new_data, new_x
        self._add_action("cut time", True)

    def average_time(self, starting_point, step, method='log', grid_dense=5):
        """
        Average time points collected (rows). This function can be use to
        average time points. Useful in multiprobe time-resolved experiments or
        flash-photolysis experiments recorded with a Photo multiplier tube where
        the number of time points is very long and are equally spaced.
        (The function assumes time vector is sorted from low to high values)


        Parameters
        ----------
        starting_point: int or float
          time points higher than this the function will be applied

        step: int, float or None
          step to consider for averaging data points

        method: 'log' or 'constant' (default: 'log')
            If constant: after starting_point the the function will return
            average time points between the step.

            If log the firsts step is step/grid_dense and the following points
            are (step/grid_dense)*n where n is point number

        grid_dense: int or float higher than 1 (default: 5)
            density of the log grid that will be applied. To high values will
            not have effect if: start_point + step/grid_dense is lower than the
            difference between the first two consecutive points higher than
            start_point. The higher the value the higher the grid dense will be.
            return.
            (only applicable if method is 'log')

        e.g.:
            time [1,2,3,4 .... 70,71,72,73,74]
            step = 10
            starting_point = 5
            method 'constant'
                time points return are [1,2,3,4,5,10,20,30,40...]
            method 'log'
                time points return are [1,2,3,4,5,6,9,14,21,30,41,54,67.5]
        """
        self._add_to_data_set("before_average_time")
        new_data, new_x = Preprocessing.average_time_points(self.data, self.x,
                                                            starting_point,
                                                            step, method,
                                                            grid_dense)
        self.data, self.x = new_data, new_x
        self._add_action("average time", True)

    def derivate_data(self, window_length=25, polyorder=3,
                      deriv=1, mode='mirror'):
        """
        Apply a Savitky-Golay filter to the data in the spectral range (rows).
        After the Savitky-Golay filter the data can be derivate which can be
        used to correct for baseline fluctuations and still perform a global
        fit or a single fit to obtain the decay times.

        Uses scipy.signal.savgol_filter
        (check scipy documentation for more information)


        Parameters
        ----------
        window_length: odd int value (default: 25)
            length defining the points for polynomial fitting

        polyorder: int or float (default: 3)
          order of the polynomial fit

        deriv: int, float or None (default: 1)
          order of the derivative after fitting

        mode: (default: 'mirror')
            mode to evaluate bounders after derivation, check
            scipy.signal.savgol_filter for the different options
        """
        self._add_to_data_set("before_derivate_data")
        new_data = Preprocessing.derivate_data(self.data, window_length,
                                               polyorder, deriv, mode)
        self.data = new_data
        self._add_action("derivate data", True)

    def cut_wavelength(self, mini=None, maxi=None, innerdata=None):
        """
        Cut columns of the data set and wavelength vector according to the
        closest values of mini and maxi margins given.
        (The function assumes column vector is sorted from low to high values)

        Parameters
        ----------
        mini: int, float or None (default: None)
          data higher than this value is kept

        maxi: int, float or None (default: None)
          data lower than this value is kept

        innerdata: cut or select (default: None)
            Only need if both mini and right maxi are given
            indicates if data inside the mini and maxi limits should be cut or
            selected.
        """
        self._add_to_data_set("before_cut_wavelength")
        new_data, new_wave = Preprocessing.cut_columns(self.data,
                                                       self.wavelength,
                                                       mini, maxi, innerdata)
        # no need to work on selected data set
        self.data, self.wavelength = new_data, new_wave
        self._add_action("cut wavelength", True)

    def del_points(self, points, dimension='time'):
        """
        Delete rows or columns from the data set according to the closest values
        given in points to the values found in dimension_vector. The length of
        dimension_vector should be equivalent to one of the data dimensions.

        Notice that the function will automatically detect the dimension of the
        delete rows or columns in data if any of their dimensions is equal to
        the length of the dimension_vector. In case that both dimensions are the
        same the axis should be given, by default is 0, which is equivalent to
        the time dimension.

        i.e.:
        points = [3,5]
        dimension_vector = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6]
        len(dimension_vector) >>> 8
        data.shape >>> (8, 10)

        Parameters
        ----------
        points: int, list or None
            estimate values of time, the closes values of dimension_vector to t
            he points given will be deleted

        dimension: str (default "time")
                can be "wavelength" or "time" indicate where points should be
                deleted
        """
        self._add_to_data_set("before_del_points")
        if dimension == 'time':
            new_data, new_x = Preprocessing.del_points(points, self.data,
                                                       self.x, 0)
            self.data, self.x = new_data, new_x
            self._add_action(f"delete point {dimension}", True)
        elif dimension == 'wavelength':
            new_data, new_wave = Preprocessing.del_points(points, self.data,
                                                          self.wavelength, 1)
            # no need to work on selected data set
            self.data, self.wavelength = new_data, new_wave
            self._add_action(f"delete point {dimension}", True)
        else:
            msg = 'dimension should be "time" or "wavelength"'
            raise ExperimentException(msg)

    def shit_time(self, value):
        """
        Shift the time vector by a value


        Parameters
        ----------
        value: int or float
            value shifting the time vector
        """
        self._add_to_data_set("before_shift_time")
        self.x = self.x - value
        self._add_action("shift time")

    def define_weights(self, rango, typo='constant', val=5):
        """
        Defines a an array that can be apply  in global fit functions as weights.
        The weights can be use to define areas where the minimizing functions is
        not reaching a good results, or to define areas that are more important
        than others in the fit. The fit with weights can be inspect as any other
        fit with the residual plot. A small constant value is generally enough
        to achieve the desire goal.

        Parameters
        ----------
        rango: list (length 2)
            list containing initial and final time values of the range
            where the weights will be applied

        typo: str (constant, exponential, r_exponential or exp_mix)
            defines the type of weighting vector returned

            constant: constant weighting value in the range
            exponential: the weighting value increase exponentially
            r_exponential: the weighting value decrease exponentially
            mix_exp: the weighting value increase and then decrease
            exponentially

            example:
            ----------
                constant value 5, [1,1,1,1,...5,5,5,5,5,....1,1,1,1,1]
                exponential for val= 2 [1,1,1,1,....2,4,9,16,25,....,1,1,1,]
                        for val= 3 [1,1,1,1,....3,8,27,64,125,....,1,1,1,]
                r_exponential [1,1,1,1,...25,16,9,4,2,...1,1,1,]
                exp_mix [1,1,1,1,...2,4,9,4,2,...1,1,1,]

        val: int (default 5)
            value for defining the weights
        """
        self.weights = define_weights(self.x, rango, typo=typo, val=val)
        self._add_action("define weights")

    def createNewDir(self, path):
        # Probably will be removed
        if not os.path.exists(path):
            os.makedirs(path)
        self.working_directory = path
        self.save['path'] = self.working_directory

    def initialize_exp_params(self, t0, fwhm, *taus, tau_inf=1E12,
                              opt_fwhm=False, vary_t0=True, global_t0=True):
        """
        function to initialize parameters for global fitting

        Parameters
        ----------
        t0: int or float
            the t0 for the fitting

        fwhm: float or None
            FWHM of the the laser pulse use in the experiment
            If None. the deconvolution parameters will not be added

        taus: int or float
            initial estimations of the decay times

        tau_inf: int or float (default 1E12)
            allows to add a constant decay value to the parameters.
            This constant modelled photoproducts formation with long decay times
            If None tau_inf is not added.
            (only applicable if fwhm is given)

        opt_fwhm: bool (default False)
            allows to optimized the FWHM.
            Theoretically this should be measured externally and be fix
            (only applicable if fwhm is given)

        vary_t0: bool (default False)
            allows to optimized the t0.
            We recommend to always set it True
            (only applicable if fwhm is given)

        global_t0: bool (default True)
            Important: only applicable if fwhm is given and data is chirp
            corrected. Allows to fit the t0 globally (setting True), which is
            faster. In case this first option fit does not give good results
            in the short time scale the t0 can be independently fitted (slower)
            (setting False) which may give better results.
        """
        taus = list(taus)
        self._last_params = {'t0': t0, 'fwhm': fwhm, 'taus': taus,
                             'tau_inf': tau_inf, 'opt_fwhm': opt_fwhm}
        self._exp_no = len(taus)
        param_creator = GlobExpParameters(self.selected_traces.shape[1], taus)
        if fwhm is None:
            self._deconv = False
            vary_t0 = False
            correction = False
        else:
            if global_t0 and not self.GVD_corrected:
                correction = False
            elif not global_t0:
                correction = False
            else:
                correction = True
            self._deconv = True
            self._tau_inf = tau_inf
        param_creator.adjustParams(t0, vary_t0, fwhm, opt_fwhm, correction, tau_inf)
        self.params = param_creator.params
        self._params_initialized = 'Exponential'
        self._add_action(f'new {self._params_initialized} parameters initialized')

    def initialize_target_params(self, t0, fwhm, *taus, tau_inf=1E12, opt_fwhm=False):
        pass

    def global_fit(self, vary=True, maxfev=5000, apply_weights=False):
        """
        Perform a exponential or a target global fits to the selected traces.
        The type of fits depends on the parameters initialized.

        Parameters
        ----------
        vary: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should
            be a list of bool equal with len equal to the number of taus. Each
            entry defines if a initial taus should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        apply_weights: bool (default False)
            If True and weights have been defined, this will be applied in the
            fit (for defining weights) check the function define_weights.
        """
        if self._params_initialized == 'Exponential':
            minimizer = GlobalFitExponential(self.x, self.selected_traces, self._exp_no, self.params,
                                             self._deconv, self._tau_inf, GVD_corrected=self.GVD_corrected)
        elif self._params_initialized == 'Target':
            minimizer = GlobalFitTarget(self.selected_traces, self.selected_wavelength, self.params)
        else:
            raise ExperimentException('Parameters need to be initiliazed first"')
        if apply_weights:
            minimizer.weights = self.weights
        results = minimizer.global_fit(vary, maxfev, apply_weights)
        results.details['svd_fit'] = self._SVD_fit
        results.wavelength = self.selected_wavelength
        results.details['avg_traces'] = self._averige_selected_traces
        self._fit_number += 1
        self.fit_records.global_fits[self._fit_number] = results
        self._add_action(f'{self._params_initialized} fit performed')
        self._update_last_params(results.params)

    def single_exp_fit(self, wave, average, t0, fwhm, *taus, vary=True, tau_inf=1E12, maxfev=5000,
                       apply_weights=False, opt_fwhm=False, plot=True):
        """
        Perform an exponential fit to a single trace

        Parameters
        ----------
        wave: int or float
            the closest value in the wavelength vector traces will be selected.

        average: int (default 1)
            Binning points surrounding the selected wavelengths.
            e. g.: if point is 1 trace = mean(index-1, index, index+1)
        t0: int or float
            the t0 for the fitting

        fwhm: float or None
            FWHM of the the laser pulse use in the experiment
            If None. the deconvolution parameters will not be added

        taus: int or float
            initial estimations of the decay times

        tau_inf: int or float (default 1E12)
            allows to add a constant decay value to the parameters.
            This modelled photoproducts formation with long decay times
            If None tau_inf is not added.
            (only applicable if fwhm is given)

        opt_fwhm: bool (default False)
            allows to optimized the FWHM.
            Theoretically this should be measured externally and be fix
            (only applicable if fwhm is given)

        vary: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should
            be a list of bool equal with len equal to the number of taus.
            Each entry defines if a initial taus should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        apply_weights: bool (default False)
            If True and weights have been defined, this will be applied in the
            fit (for defining weights) check the function define_weights.

        plot: bool (default True)
            If True the results are automatically plotted
        """
        taus = list(taus)
        print(taus)
        trace, wave = select_traces(self.data, self.wavelength, [wave], average)
        results = self._one_trace_fit(trace, t0, fwhm, *taus, vary=vary,
                                      tau_inf=tau_inf, maxfev=maxfev,
                                      apply_weights=apply_weights,
                                      opt_fwhm=opt_fwhm)
        results.wavelenght = wave
        key = len(self.fit_records.single_fits) + 1
        self.fit_records.single_fits[key] = results
        self._add_action('Exponential single fit performed')
        if plot:
            self.plot_single_fit(key)

    def integral_band_exp_fit(self, wave_range: list, t0, fwhm, *taus, vary=True, tau_inf=1E12, maxfev=5000,
                              apply_weights=False, opt_fwhm=False, plot=True):
        """
        Perform an exponential fit to an integrated are of the spectral range of the data set.
        This type of fits allows for example to identify time constants attributed to cooling
        since the integration compensate the effects and the contribution of this type of
        phenomena to the decay decreases or disappears.

        Parameters
        ----------
        wave_range: list (lenght 2) or float
            The area between the two entries of the wavelength range is integrated and fitted.

        t0: int or float
            the t0 for the fitting

        fwhm: float or None
            FWHM of the the laser pulse use in the experiment
            If None. the deconvolution parameters will not be added

        taus: int or float
            initial estimations of the decay times

        tau_inf: int or float (default 1E12)
            allows to add a constant decay value to the parameters.
            This modelled photoproducts formation with long decay times
            If None tau_inf is not added.
            (only applicable if fwhm is given)

        opt_fwhm: bool (default False)
            allows to optimized the FWHM.
            Theoretically this should be measured externally and be fix
            (only applicable if fwhm is given)

        vary: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should be a list of bool
            equal with len equal to the number of taus. Each entry defines if a initial taus
            should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        apply_weights: bool (default False)
            If True and weights have been defined, this will be applied in the fit (for defining weights) check
            the function define_weights.

        plot: bool (default True)
            If True the results are automatically plotted
        """
        taus = list(taus)
        indexes = [np.argmax(abs(self.wavelength-wave_range[0])), np.argmax(abs(self.wavelength-wave_range[1]))]
        trace = np.array([np.trapz(self.data[i,indexes[0]:indexes[1]], x=self.wavelength[indexes[0]:indexes[1]])
                          for i in range(len(self.data))])
        results = self._one_trace_fit(trace, t0, fwhm, *taus, vary=vary, tau_inf=tau_inf, maxfev=maxfev,
                                      apply_weights=apply_weights, opt_fwhm=opt_fwhm)
        results.details['integral band'] = wave_range
        key = len(self.fit_records.integral_band_fits) + 1
        self.fit_records.single_fits[key] = results
        self._add_action(f'Integral band fit between {wave_range[0]} and {wave_range[1]} performed')
        if plot:
            self.plot_integral_band_fit(key)

    def _one_trace_fit(self, trace, t0, fwhm, *taus, vary=True, tau_inf=1E12, maxfev=5000,
                       apply_weights=False, opt_fwhm=False):
        """
        Real fitting function used by "integral_band_exp_fit" and "single_exp_fit"
        """
        print(taus)
        param_creator = GlobExpParameters(1, taus)
        param_creator.adjustParams(t0, vary, fwhm, opt_fwhm, self.GVD_corrected, tau_inf)
        print(param_creator.params)
        deconv = True if fwhm is not None else False
        minimizer = GlobalFitExponential(self.x, trace, len(taus), params=param_creator.params,
                                         deconv=deconv, tau_inf=tau_inf, GVD_corrected=False)
        results = minimizer.global_fit(vary, maxfev, apply_weights)
        return results

    def plot_single_fit(self, fit_number=None, details=True):
        """
        Function that generates a figure with the results of the fit stored in
        the single_fits

        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results single_fits dictionary. If
            None the last fit in  will e considered

        details: bool (default True)
            If True the decay times obtained in the fit are included in the
            figure
        """
        if fit_number in self.fit_records.single_fits.keys():
            return self._plot_single_trace_fit(self.fit_records.single_fits,
                                               fit_number, details)
        else:
            msg = 'Fit number not in records'
            raise ExperimentException(msg)

    def plot_integral_band_fit(self, fit_number=None, details=True):
        """
        Function that generates a figure with the results of the fit stored in
        the integral_band_fits

        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results integral_band_fits dictionary.
            If None the last fit in  will be considered

        details: bool (default True)
            If True the decay times obtained in the fit are included in the
            figure
        """
        if fit_number is None:
            fit_number = len(self.fit_records.integral_band_fits)
        if fit_number in self.fit_records.integral_band_fits.keys():
            fig, ax = self._plot_single_trace_fit(self.fit_records.integral_band_fits, fit_number, details)
            rang = self.fit_records.integral_band_fits[fit_number].details['integral band']
            w_unit = 'cm$^{-1}$' if self._units['wavelength_unit']  == 'cm-1' else self._units['wavelength_unit']
            ax[1].legend(['_', f'Integral band {rang[0]}-{rang[1]} {w_unit}'])
            return fig, ax
        else:
            msg = 'Fit number not in records'
            raise ExperimentException(msg)

    def _plot_single_trace_fit (self, container, fit_number, details):
        """
        Base plot function used by "plot_integral_band_fit" and "plot_single_fit"
        """
        plotter = ExploreResults(container[fit_number], **self._units)
        _, _, _, params, exp_no, deconv, tau_inf, _, _, _ = plotter._get_values(fit_number=fit_number)
        fig, ax = plotter.plot_fit()
        if details:
            testx = plotter._legend_plot_DAS(params, exp_no, deconv, tau_inf, 'Exponential', 2)
            textstr = '\n'.join(testx)
            texto = AnchoredText(s=textstr, loc=9)
            ax[1].add_artist(texto)
        return fig, ax

    def refit_with_SVD_fit_result(self, fit_number=None, fit_data='all'):
        """
        Not finished
        """
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        if fit_data == 'all':
            data_fit = self.selected_traces
        else:
            data_fit = self.data
        if type_fit == 'Exponential':
            taus = [params['tau%i_1' % (i+1)].value for i in range(exp_no)]
            t0 = params['t0_1'].value
            if deconv:
                fwhm = params['fwhm_1'].value
            param_creator = GlobExpParameters(data_fit.shape[1], taus)
            param_creator.adjustParams(t0, True, fwhm, False, self.GVD_corrected, tau_inf)
            params_fit = param_creator.params
            minimizer = GlobalFitExponential(self.x, self.selected_traces, exp_no, params_fit,
                                             deconv, tau_inf, False)
            minimizer.pre_fit()
        ## Todo

        pass

    def select_traces(self, points=10, average=1, avoid_regions=None):
        """
        Method to select traces from the data attribute and defines a subset of traces
        in the selected_traces attribute. If the parameters have been initialize automatically
        re-adapts them to the new selected traces.
        (The function assumes wavelength vector is sorted from low to high values)

        Parameters
        ----------
        points: int or list or "auto" (default 10)
            If type(space) =int: a series of traces separated by the value indicated
            will be selected.
            If type(space) = list: the traces in the list will be selected.
            If space = auto, the number of returned traces is 10 and equally spaced
            along the wavelength vector and points is set to 0

        average: int (default 1)
            Binning points surrounding the selected wavelengths.
            e. g.: if point is 1 trace = mean(index-1, index, index+1)

        avoid_regions: list of list (default None)
            Defines wavelength regions that are avoided in the selection when space
            is an integer. The sub_list should have two elements defining the region
            to avoid in wavelength values
            i. e.: [[380,450],[520,530] traces with wavelength values between 380-450
                   and 520-530 will not be selected
        """
        super().select_traces(points, average, avoid_regions)
        self._readapt_params()
        self._averige_selected_traces = average if points != 'all' else 0
        self._add_action("Selected traces")

    def select_region(self, mini, maxi):
        """
        Select a region of the data as selected traces according to the closest
        values of mini and maxi to the wavelength vector. If the parameters have
        been initialize automatically re-adapts them to the new selected traces
        (The function assumes wavelength vector is sorted from low to high values)

        Parameters
        ----------
        mini: int, float or None
          data higher than this value is kept

        maxi: int, float or None
          data lower than this value is kept
        """
        new_data, new_wave = Preprocessing.cut_columns(self.data, self.wavelength, mini, maxi, True)
        self.selected_traces, self.selected_wavelength = new_data, new_wave
        self._readapt_params()
        self._averige_selected_traces = 0
        self._add_action("Selected region as traces")

    def _update_last_params(self, params):
        """
        Function updating parameters after a global fit
        """
        if self._params_initialized == 'Exponential':
            self._last_params['t0'] = params['t0_1'].value
            self._last_params['taus'] = [params['tau%i_1' % (i + 1)].value for i in range(self._exp_no)]
        elif self._params_initialized == 'Target':
            ## todo
            pass
        else:
            pass

    def restore_data(self, action: str):
        """
        Restore the data to a point previous to a preprocesing action
        actions should be the name of the function.
        e.g.: "baseline substraction" or "baseline_substraction"
        Possible action:
            baseline_substraction
            average_time
            cut_time
            cut_wavelength
            del_points
            derivate_data
            shift_time
            subtract_polynomial_baseline
            correct_chirp/correct_GVD
        """
        action = '_'.join(action.split(' '))
        key = [i for i in self.data_sets.__dict__.keys() if action in i]
        msg = f'data has not been {action}'
        if len(key) == 1:
            key = key[0]
        elif len(key) >= 1:
            msg = f'{action} is not ambiguous specify time or wave'
        else:
            key = 'Not_an_action'
        if hasattr(self.data_sets, key):
            container = getattr(self.data_sets, key)
            self.data = container.data
            self.x = container.time
            self.wavelength = container.wavelength
            self.selected_traces = container.data
            self.selected_wavelength = container.wavelength
            self.preprocessing_report = copy.copy(container.report)
            self.preprocessing_report._last_action = None
            msg = ' '.join(key.split('_'))
            self._add_action(f'restore {msg}')
        else:
            raise ExperimentException(msg)

    def undo_last_preprocesing(self):
        """
        Undo the last preprocesing action perform to the data
        """
        if self.preprocessing_report._last_action is None:
            print('No preprocesing action run or already undo last action')
        else:
            key = self.preprocessing_report._last_action
            self.restore_data(key)
            self.preprocessing_report._last_action = None

    def _add_to_data_set(self, key):
        """
        add data to data sets after a preprocesing action
        """
        if hasattr(self.data_sets, key):
            pass
        else:
            report = copy.copy(self.preprocessing_report)
            container = UnvariableContainer(x=self.x, data=self.data, wavelength=self.wavelength,
                                            report=report)
            self.preprocessing_report._last_action = key
            self.data_sets.__setattr__(key, container)
            self._last_data_sets = container

    def _add_action(self, value, re_select_traces=False):
        """
        add action to action records
        """
        val = len(self.action_records.__dict__)-2
        self.action_records.__setattr__(f"_{val}", value)
        if re_select_traces:
            self._re_select_traces()

    def _re_select_traces(self):
        if self._SVD_fit:
            self._calculateSVD()
            self.select_SVD_vectors(self.selected_traces.shape[1])
        else:
            avg = self._averige_selected_traces
            trace, wave = self.select_traces(self.selected_traces, avg)
            self.selected_traces, self.selected_wavelength = trace, wave
            val = len(self.action_records.__dict__) - 3
            delattr(self.action_records, f"_{val}" )

    def _readapt_params(self):
        """
        Function to automatically re-adapt parameters to a new selection of data sets
        """
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
            t0 = self._last_params['t0']
            fwhm = self._last_params['fwhm']
            self.initialize_target_params(t0, fwhm,)
            # self.initialize_target_params()
        else:
            pass