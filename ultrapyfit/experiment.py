# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:55:03 2020

@author: lucas
"""
import numpy as np
from lmfit import Parameters
import datetime
from ultrapyfit.graphics.ExploreResults import ExploreResults
from ultrapyfit.graphics.ExploreData import ExploreData
from ultrapyfit.fit.GlobalParams import GlobExpParameters, GlobalTargetParameters
from ultrapyfit.fit.GlobalFitBootstrap import BootStrap
from ultrapyfit.utils.ChirpCorrection import EstimationGVDPolynom, EstimationGVDSellmeier
from ultrapyfit.utils.divers import define_weights, UnvariableContainer, LabBook,\
    book_annotate, read_data, TimeUnitFormater, select_traces
from ultrapyfit.fit.targetmodel import ModelWindow
from ultrapyfit.utils.Preprocessing import ExperimentException
from ultrapyfit.utils.Preprocessing import Preprocessing as Prep
from ultrapyfit.fit.GlobalFit import GlobalFitExponential, GlobalFitTarget
import os
from matplotlib.offsetbox import AnchoredText
import copy
import pickle
import sys


class SaveExperiment:
    # TODO updat to new structure
    """
    Class that extract and save important features of an Experiment instance
    and saved them as a dictionary that can be later on reload and used.
    This class is directly used when you run the Experiment.save() method.
    It also directly save the Experiment pass after instantiation.
    
    Attributes
    ----------
    path: string
        path containing the name and extension that should be used to save the 
        data.
        
    experiment: Experiment instance
        An instance of Experiment class that want to be saved.
        
    save_object: dictionary
        Object that will be pcikle
    """
    def __init__(self, path, experiment, auto_save=True):
        self.path = path
        self.experiment = experiment
        self.save_object = {}
        if auto_save:
            self.save()

    def save(self):
        """
        Save the save_object dictionary as pickle object 
        """
        self._extract_objects()
        path = self.path
        if not path[:-4] == '.exp':
            path += '.exp'
        with open(path, 'wb') as file:
            pickle.dump(self.save_object, file,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def _extract_objects(self):
        """
        Extract the features of an Experiment instance and add to the 
        save_object dictionary
        """
        details = {'units': self.experiment._units,
                   'GVD': self.experiment.preprocessing.GVD_corrected,
                   'excitation': self.experiment.excitation,
                   'path': self.experiment._data_path,
                   'deconv': self.experiment.fitting._deconv,
                   'n_fits': self.experiment.fitting._fit_number}

        self.save_object["report"] = self.experiment.preprocessing.report
        self.save_object['fits'] = self.experiment.fitting.fit_records
        self.save_object['actions'] = self.experiment._action_records
        self.save_object['datas'] = self.experiment.preprocessing.data_sets
        self.save_object['data'] = self.experiment.data
        self.save_object['x'] = self.experiment.x
        self.save_object['wavelength'] = self.experiment.wavelength
        self.save_object['detail'] = details


class Experiment(ExploreData):
    """
    Class to work with a time resolved data sets. To easily preprocess the data,
    obtain quality fits and result. Is possible to explore the data set and keep
    track of actions done. Finally to easily create figures already formatted.
    The class inherits ExploreData and ExploreResults therefore all methods for
    plotting and exploring a data set, explore the SVD space and check results
    are available.

    Attributes
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2darray
        Array containing the data, the number of rows should be equal to the
        len(x) and the number of columns equal to len(wavelength)

    wavelength: 1darray
        Wavelength vector

    selected_traces: 2darray
        Sub dataset of data. The global fits are performed only in the selected
        traces data set. Preprocesing actions are done on both data and selected
        traces. To select a part of the data set, use select_traces,
        select_traces_graph. To select the SVD left values for fitting use
        select_SVD_vectors or plot_SVD(1, select=True).

    selected_wavelength: 1darray
        Sub dataset of wavelength. Automatically updated when selected traces
        is update

    time_unit: str (default ps)
        Contains the strings of the time units to format axis labels and 
        legends automatically:
        time_unit str of the unit. >>> e.g.: 'ps' for picosecond
        Can be passes as kwargs when instantiating the object

    wavelength_unit: str (default nm)
        Contains the strings of the wavelength units to format axis labels and
        legends automatically:
        wavelength_unit str of the unit: >>> e.g.: 'nm' for nanometers
        Can be passes as kwargs when instantiating the object

    params: lmfit parameters object
        object containing the initial parameters values used to build an
        exponential model. These parameters are iteratively optimize to reduce
        the residual matrix formed by data-model (error matrix) using
        Levenberg-Marquardt algorithm. After a global fit they are actualized
        to the results obtained. After selection of new traces, this are
        automatically re-adapted.

    weights: dict
        contains the weigthing vector that will be apply if apply_weights is 
        set to True in any of the fitting functions. The weight can be define 
        with the define weights function.

    GVD_corrected/chirp_corrected: (default False)
        Indicates if the data has been chirp corrected. It set to True after
        calling the chrip_correction method.

    preprocesing.report: class LabBook
        Object containing and keeping track of the preprocessing actions done 
        to the data set. The preprocesing.report.print() method will print the
        status of the actions done and parameters passed to each of the
        preprocesing functions. The general report method of the
        Experiment class will also print it.

    action_records: class UnvariableContainer
        Object containing and keeping track of the important actions done.
        similar to the preprocesing.report t content can be printed.
        The general report method of the Experiment class will also print it.

    fit_records: class UnvariableContainer
        Object containing and keeping track of fit results done. The attributes
        are: single_fits, global_fits, integral_band_fits, bootstrap_record,
        conf_interval and target_models

    excitation: float or int (default None)
        If given the spectra figures will display a white box ± 10 units of
        excitation automatically to cover the excitation if this value is
        between the wavelength range plotted. This value is needed for some
        chirp correction methods. Note that the excitation is not needed for
        fitting the data or any other method.
    """
    def __init__(self, x, data, wavelength=None, path=None, **kwargs):
        """
        Constructor function to initialize the Experiment class


        Parameters
        ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            array containing the data, the number of rows should be equal to
            the len(x)

        wavelength: 1darray (default None)
            Wavelength vector. Alternatively the wavelength can be "None" and
            may be later on calibrated. If None, wavelength vector will be an
            array from 0 to the length of data columns. Although it can be
            initialize without wavelength vector, is recommended to pass this
            value or calibrate in a first step.

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
        self.excitation = None
        # self.GVD_corrected = False
        self._action_records = UnvariableContainer(name="Sequence of actions")
        load = datetime.datetime.now().strftime(
            "day: %d %b %Y | hour: %H:%M:%S")
        self._add_action("Created experiment " + load)
        # self.preprocessing.report = LabBook(name="Pre-processing")
        self._data_path = path
        self._units = units
        self._average_selected_traces = 0
        self._unit_formater = TimeUnitFormater(self._units['time_unit'])
        self._silent_selection_of_traces = False
        # _fit_number take record of global exponential and target fits ran.
        self.preprocessing = self._Preprocessing(self)
        self.fitting = self._Fit(self)
        ExploreData.__init__(self, self.x, self.data, self.wavelength,
                             self.selected_traces, self.selected_wavelength,
                             'viridis', **self._units)

    """
    Properties and structural functions
    """
    def get_action_records(self):
        return self._action_records.get_protected_attributes()

    def print_action_records(self):
        self._action_records.print(False, True, True)

    @property
    def time(self):
        return self.x

    @time.setter
    def time(self, time):
        self.x = time

    @classmethod
    def load_data(cls, path: str, wavelength=0, time=0, wave_is_row=False,
                  separator=',', decimal='.'):
        """
        Load data from a file and return dn Experiment instance

        Parameters
        ----------
        path: str
            path to the data file

        wavelength: int (default 0)
            defines the element where to find the wavelength vector in its
            direction which is defined by wave_is_row parameter.
            i.e.: if wavelength correspond to columns, wavelength=0 indicates
            is the first column of the data file if wavelength correspond to
            rows, then wavelength=0 is first row

        time: int (default 0)
            defines the element where to find the time vector in its direction
            which is defined by wave_is_row parameter. i.e.: if times correspond
            to columns, time=0 indicates is the first column of the data file

        wave_is_row: bool (default False)
            defines if in the original data set the wavelength correspond
            to the rows.

        separator: str (default ',')
            defines the separator in the data (any value that can be used in
            pandas read_csv is valid. For tab uses \t

        decimal: int (default '.')
            defines the decimal point in the data

        Returns
        ----------
        Experiment instance
        """
        try:
            x, data, wave = read_data(path, wavelength, time, wave_is_row,
                                      separator, decimal)
        except Exception as exception:
            raise ExperimentException(exception)
        else:
            experiment = cls(x, data, wave, path)
            experiment.preprocessing.report.loaded_file = path
        return experiment

    @classmethod
    def load(cls, path: str):
        """
        Load a saved Experiment
        """
        error = False
        file_found = False
        unpickle = False
        instantiate = False
        try:
            with open(path, 'rb') as file:
                file_found = True
                object_load = pickle.load(file)
                unpickle = True
                x = object_load['x']
                data = object_load['data']
                wavelength = object_load['wavelength']
                experiment = cls(x, data, wavelength)
                # updatae preprocesing so that annoatate in the loaded labBook
                experiment.preprocessing = cls._Preprocessing(experiment,
                                                              object_load["report"])
                instantiate = True
                experiment.fitting.fit_records = object_load['fits']
                experiment.fitting._fits = object_load['fits'].global_fits
                experiment._action_records = object_load['actions']
                experiment.preprocessing.data_sets = object_load['datas']
                experiment._units = object_load['detail']['units']
                gvd = object_load['detail']['GVD']
                experiment.preprocessing.GVD_corrected = gvd
                experiment.excitation = object_load['detail']['excitation']
                experiment._data_path = object_load['detail']['path']
                experiment.fitting._deconv = object_load['detail']['deconv']
                fit_number = object_load['detail']['n_fits']
                experiment.fitting._fit_number = fit_number
                load = datetime.datetime.now().strftime(
                    "day: %d %b %Y | hour: %H:%M:%S")
                experiment._add_action("reload experiment "+load)

        except Exception:
            error = True
        else:
            if type(experiment) == Experiment:
                return experiment
        finally:
            if error:
                if not file_found:
                    msg = 'File not found, incorrect path'
                elif not file_found:
                    msg = 'Unable to open the specify file'
                elif not instantiate:
                    msg = 'File do not correspond to a ultrapyfit Experiment class'
                elif not unpickle:
                    msg = 'Unable to unpickle the specified file'
                else:
                    msg = 'Undefined error occur while loading the file'
                raise ExperimentException(msg)
            else:
                print('Experiment load successfully')

    def load_fit(self, path):
        # TODO load a fit result
        pass

    def save(self, path):
        """
        Saved the current Experiment

        Parameters
        ----------
        path: string
            path where to save the Experiment
        """
        SaveExperiment(path, self)

    # def createNewDir(self, path):
    #    # Probably will be removed; for the moment is not working
    #     if not os.path.exists(path):
    #        os.makedirs(path)
    #     self.working_directory = path
    #     self.save['path'] = self.working_directory

    """
    Check status functions
    """
    def describe_data(self):
        """
        Print description of the data
        """
        name = 'data description'
        print(f'\t {name}')
        print('-'*(len(name)+10))
        print('\tData set\t\tNº traces\tNº spectra')
        name = ['Loaded data', 'Current data']
        for ii, i in enumerate([self.preprocessing.data_sets.original_data.data,
                                self.data]):
            print(f'\t{name[ii]}:\t {i.shape[1]}\t\t\t{i.shape[0]}')
        print(f'\tSelected traces: {self.selected_traces.shape[1]}')
        print('\n\t units')
        print('-' * (len('units') + 10))
        print(f'\tTime unit: {self.time_unit}')
        print(f'\tWavelength unit: {self.wavelength_unit}')

    def print_general_report(self, output_file=None):
        """
        Print the general report of the experiment

        Parameters
        ----------
        output_file: str or None (default None)
            give the output file directory where the report will be printed
        """
        def printing():
            self.describe_data()
            print('============================================\n')
            self.preprocessing.report.print()
            print('============================================\n')
            for i in range(len(self.fitting.fit_records.global_fits)):
                self.fitting.print_fit_results(i + 1)
            print('============================================\n')
            self._action_records.print(False, True, True)

        if output_file is not None:
            total_path = os.path.abspath(output_file)
            try:
                path, extension = os.path.splitext(total_path)
                original_stdout = sys.stdout
                if extension == "txt":
                    pass
                else:
                    total_path = path + ".txt"
                with open(total_path, "w") as file:
                    sys.stdout = file
                    printing()
                    sys.stdout = original_stdout
            except Exception as e:
                msg = f"Unable to save in '{total_path}'"
                raise ExperimentException(msg)
            else:
                msg = f"Report save in '{total_path}'"
                print(msg)
        else:
            printing()

    """
    Preprocessing functions
    """
    class _Preprocessing:
        """
        Class that aggregate all preprocessing functions under the name
        Preprocessing. Therefore, the functions should be used as follow:

        experiment = Experiment(time, data, wavelength) #create an instance
        experiment.Preprocessing.function(*arg,*+kwargs) #preprocess function
        """
        def __init__(self, experiment, report=None):
            self._experiment = experiment
            # self.x = x
            # self.data = data
            # self.wavelength = wavelength
            if report is None:
                self.report = LabBook(name="Pre-processing")
            else:
                self.report = report
            self._chirp_corrector = None
            self._last_data_sets = None
            self.GVD_corrected = False
            self.data_sets = UnvariableContainer()
            self.data_sets.original_data = UnvariableContainer(time=self._experiment.x,
                                                               data=self._experiment.data,
                                                               wavelength=self._experiment.wavelength)
            self._final_initialization()

        def _final_initialization(self):
            # modified functions to the decorator book anotate
            self.baseline_substraction = book_annotate(
                self.report)(self.baseline_substraction)
            self.subtract_polynomial_baseline = book_annotate(
                self.report)(self.subtract_polynomial_baseline)
            self.cut_time = book_annotate(self.report)(self.cut_time)
            self.average_time = book_annotate(self.report)(self.average_time)
            self.derivate_data = book_annotate(self.report)(self.derivate_data)
            self.calibrate_wavelength = book_annotate(
                self.report)(self.calibrate_wavelength)
            self.cut_wavelength = book_annotate(self.report)(self.cut_wavelength)
            self.delete_points = book_annotate(self.report)(self.delete_points)
            self.shift_time = book_annotate(self.report)(self.shift_time)

        @property
        def chirp_corrected(self):
            return self.GVD_corrected

        @chirp_corrected.setter
        def chirp_corrected(self, value):
            if type(value) == bool:
                self.GVD_corrected = value

        def chirp_correction_graphically(self, method, excitation=None):
            """
            Function to correct the chrip or GVD dispersion graphically.

            Parameters
            ----------

            method: str (valid strings "sellmeier"; "polynomial")
                defines the method use, either using the Sellmeier equation or
                fitting a polynomial.

            excitation: float (default None)
                give the excitation uses in the experiment; only needed if the
                method is "sellmeier"
            """
            func = self._change_data_after_chrip_correction
            if method == 'sellmeier':
                if excitation is None:
                    msg = 'The excitation must be defined'
                    raise ExperimentException(msg)
                self._chirp_corrector = EstimationGVDSellmeier(self._experiment.x,
                                                               self._experiment.data,
                                                               self._experiment.wavelength,
                                                               excitation,
                                                               function=func)
            elif method == 'polynomial':
                self._chirp_corrector = EstimationGVDPolynom(self._experiment.x,
                                                             self._experiment.data,
                                                             self._experiment.wavelength,
                                                             function=func)
            else:
                msg = 'Method can only be "sellmeier" or "polynomial"'
                raise ExperimentException(msg)
            self._chirp_corrector.estimate_GVD_from_grath()
            # self.data = self._chirp_corrector.corrected_data
            # capture_chirp_correction(self)

        def _change_data_after_chrip_correction(self):
            """
            Internal function to update data after chrip correction
            """
            self._experiment.data = self._chirp_corrector.corrected_data
            details = self._chirp_corrector.estimation_params.details
            self.report.__setattr__('chrip_correction', details, True)
            self._experiment._add_action("correct chirp", True)
            self.GVD_corrected = True

        def calibrate_wavelength(self, pixels: list, wavelength: list,
                                 order=2):
            """
            Calibrates the wavelength vector from a set of given lists of 
            points pixels and wavelength, using a polynomial fit between the 
            point in the two list.


            Parameters
            ----------
            pixels: list
                list containing a set of values from the original array.

            wavelength: list
                list containing a set of values to which the values given in 
                the pixels list correspond in reality.

            order: int (default 2)
                Order of the polynomial use to fit pixels and wavelength.
                Notice that the order should be smaller than the len of the 
                list
            """
            self._add_to_data_set("before_calibrate_wavelength")
            new_wave = Prep.calibration_with_polynom(self._experiment.wavelength,
                                                     pixels,
                                                     wavelength,
                                                     order)

            self._experiment.wavelength = new_wave
            self._experiment._add_action("calibrate wavelength", True)

        def GVD_correction_graphically(self, method, excitation=None):
            """
            Identical to chirp_correction method
            """
            self.chirp_correction_graphically(method, excitation)

        def baseline_substraction(self, number_spec=2, only_one=False):
            """
            Subtract a initial spectrum or an average of initial spectra
            (normally before time = 0) to the entire data set.


            Parameters
            ----------
            number_spec: int or list
                This parameter defines the number of spectra to be subtracted.
                If int should be the index of the spectrum to be subtracted
                If list should be of length 2 and contain initial and final
                spectra index that want to be subtracted

            only_one: bool (default False)
                if True only the spectrum (at index number_spec) is subtracted
                if False an average from the spectrum 0 to number_spec is
                subtracted (only applicable if number_spec is an int)

            e.g.1: number_spec = [2,5] an average from spectra 2 to 5 is
            subtracted (where 2 and 5 are included)

            e.g.2: number_spec = 5 an average from spectra 0 to 5 is subtracted
                    if only_one = False; if not only spectrum 5 is subtracted
            """
            self._add_to_data_set("before_baseline_substraction")
            new_data = Prep.baseline_substraction(self._experiment.data,
                                                  number_spec=number_spec,
                                                  only_one=only_one)

            self._experiment.data = new_data
            self._experiment._add_action("baseline substraction", True)

        def subtract_polynomial_baseline(self, points, order=3):
            """
            Fit and subtract a polynomial to the data in the spectral range
            (rows). This function can be used to correct for baseline
            fluctuations typically found in time resolved IR spectroscopy.


            Parameters
            ----------
            points: list
                list containing the wavelength values where the different
                transient spectra should be zero

            order: int or float (default: 3)
               order of the polynomial fit
            """
            self._add_to_data_set("before_subtract_polynomial_baseline")
            new_data = Prep.subtract_polynomial_baseline(self._experiment.data,
                                                         self._experiment.wavelength,
                                                         points=points,
                                                         order=order)
            self._experiment.data = new_data
            self._experiment._add_action("subtract polynomial baseline", True)

        def cut_time(self, mini=None, maxi=None):
            """
            Cut time point of the data set according to the closest values of
            mini and maxi margins given to the time vector. Contrary to
            cut_wavelength function, inner cut are not available since in time
            resolved spectroscopy is not logical to cut a complete area of
            recorded times. Therefore, giving mini and maxi margins will result
             in selection of inner time values.
            (The function assumes rows vector is sorted from low to high values)


            Parameters
            ----------
            mini: int, float or None (default: None)
              data higher than this value is kept

            maxi: int, float or None (default: None)
              data lower than this value is kept
            """
            self._add_to_data_set("before_cut_time")
            new_data, new_x = Prep.cut_rows(self._experiment.data,
                                            self._experiment.x,
                                            mini,
                                            maxi)
            self._experiment.data, self._experiment.x = new_data, new_x
            self._experiment._add_action("cut time", True)

        def average_time(self, starting_point, step,
                         method='log', grid_dense=5):
            """
            Average time points collected (rows). This function can be use to
            average time points. Useful in multiprobe time-resolved experiments
            or flash-photolysis experiments recorded with a Photo multiplier
            tube where the number of time points is very long and are equally
            spaced.
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

                If log the firsts step is step/grid_dense and the following
                points are (step/grid_dense)*n where n is point number

            grid_dense: int or float higher than 1 (default: 5)
                density of the log grid that will be applied. To high values
                will not have effect if: start_point + step/grid_dense is lower
                than the difference between the first two consecutive points
                higher than start_point. The higher the value the higher the
                grid dense will be return.
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
            new_data, new_x = Prep.average_time_points(self._experiment.data,
                                                       self._experiment.x,
                                                       starting_point,
                                                       step, method,
                                                       grid_dense)
            self._experiment.data, self._experiment.x = new_data, new_x
            self._experiment._add_action("average time", True)

        def derivate_data(self, window_length=25, polyorder=3,
                          deriv=1, mode='mirror'):
            """
            Apply a Savitky-Golay filter to the data in the spectral range
            (rows). After the Savitky-Golay filter the data can be derivate
            which can be used to correct for baseline fluctuations and still
            perform a global fit or a single fit to obtain the decay times.

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
            new_data = Prep.derivate_data(self._experiment.data,
                                          window_length,
                                          polyorder, deriv, mode)
            self._experiment.data = new_data
            self._experiment._add_action("derivate data", True)

        def cut_wavelength(self, mini=None, maxi=None, innerdata=None):
            """
            Cut columns of the data set and wavelength vector according to the
            closest values of mini and maxi margins given.
            (The function assumes column vector is sorted from low to high
            values)

            Parameters
            ----------
            mini: int, float or None (default: None)
              data higher than this value is kept

            maxi: int, float or None (default: None)
              data lower than this value is kept

            innerdata: cut or select (default: None)
                Only need if both mini and right maxi are given
                indicates if data inside the mini and maxi limits should be cut
                or selected.
            """
            self._add_to_data_set("before_cut_wavelength")
            new_data, new_wave = Prep.cut_columns(self._experiment.data,
                                                  self._experiment.wavelength,
                                                  mini, maxi,
                                                  innerdata)
            # no need to work on selected data set
            self._experiment.data = new_data
            self._experiment.wavelength = new_wave
            self._experiment._add_action("cut wavelength", True)

        def delete_points(self, points, dimension='time'):
            """
            Delete rows or columns from the data set according to the closest
            values given in points to the values found in dimension_vector.
            The length of dimension_vector should be equivalent to one of the
            data dimensions.

            Notice that the function will automatically detect the dimension of
            the delete rows or columns in data if any of their dimensions is
            equal to the length of the dimension_vector. In case that both
            dimensions are the same the axis should be given, by default is 0,
             which is equivalent to the time dimension.

            i.e.:
            points = [3,5]
            dimension_vector = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6]
            len(dimension_vector) >>> 8
            data.shape >>> (8, 10)

            Parameters
            ----------
            points: int, list or None
                estimate values of time, the closes values of dimension_vector
                to the points given will be deleted

            dimension: str (default "time")
                    can be "wavelength" or "time" indicate where points should
                     be deleted
            """
            self._add_to_data_set("before_delete_points")
            if dimension == 'time':
                new_data, new_x = Prep.del_points(points,
                                                  self._experiment.data,
                                                  self._experiment.x,
                                                  0)
                self._experiment.data, self._experiment.x = new_data, new_x
                self._experiment._add_action(f"delete points {dimension}",
                                             True)
            elif dimension == 'wavelength':
                new_data, new_wave = Prep.del_points(points,
                                                     self._experiment.data,
                                                     self._experiment.wavelength,
                                                     1)
                # no need to work on selected data set
                self._experiment.data = new_data
                self._experiment.wavelength = new_wave
                self._experiment._add_action(f"delete points {dimension}",
                                             True)
            else:
                msg = 'dimension should be "time" or "wavelength"'
                raise ExperimentException(msg)

        def shift_time(self, value):
            """
            Shift the time vector by a value


            Parameters
            ----------
            value: int or float
                value shifting the time vector
            """
            self._add_to_data_set("before_shift_time")
            self._experiment.x = self._experiment.x - value
            self._experiment._add_action("shift time")

        def restore_data(self, action: str):
            """
            Restore the data to a point previous to a preprocessing action
            actions should be the name of the function.
            e.g.: "baseline substraction" or "baseline_substraction" are both 
            valid

            Parameters
            ----------
            action:
                Possible actions:
                    original_data,
                    baseline_substraction,
                    average_time,
                    cut_time,
                    cut_wavelength,
                    delete_points,
                    derivate_data,
                    shift_time,
                    subtract_polynomial_baseline,
                    correct_chirp/correct_GVD,
                    calibrate_wavelength,
            """
            action = '_'.join(action.split(' '))
            key = [i for i in self.data_sets.__dict__.keys() if action in i]
            msg = f'data has not been {action}'
            if len(key) == 1:
                key = key[0]
            elif len(key) >= 1:
                msg = f'"{action}" is ambiguous specify "cut time" or "cut wave"'
                key = 'Not_an_action'
            else:
                key = 'Not_an_action'
            if hasattr(self.data_sets, key):
                container = getattr(self.data_sets, key)
                self._experiment.data = container.data
                self._experiment.x = container.x
                self._experiment.wavelength = container.wavelength
                self._experiment.selected_traces = container.data
                self._experiment.selected_wavelength = container.wavelength
                keys = [i for i in self.report.__dict__.keys()]
                for i in keys:
                    if i not in container.report.__dict__.keys():
                        delattr(self.report, i)
                for i in container.report.__dict__.keys():
                    if i not in self.report.__dict__.keys():
                        atr = getattr(container.report, i)
                        setattr(self.report, i, atr)
                self.report._last_action = None
                write_action = ' '.join(key.split('_'))
                self._experiment._add_action(f'restore {write_action}')
            else:
                raise ExperimentException(msg)

        def undo_last_preprocesing(self):
            """
            Undo the last preprocesing action perform to the data
            """
            if self.report._last_action is None:
                print('No preprocesing action run or already undo last action')
            else:
                key = self.report._last_action
                self.restore_data(key)
                self.report._last_action = None

        def _add_to_data_set(self, key):
            """
            add data to data sets after a preprocesing action
            """
            if hasattr(self.data_sets, key):
                pass
            else:
                report = copy.copy(self.report)
                container = UnvariableContainer(x=self._experiment.x,
                                                data=self._experiment.data,
                                                wavelength=self._experiment.wavelength,
                                                report=report)
                self.report._last_action = key
                self.data_sets.__setattr__(key, container)
                self._last_data_sets = container

    """
    Parameters functions and fitting
    """
    class _Fit(ExploreResults):
        def __init__(self, experiment):
            self._experiment = experiment
            self._params_initialized = False
            self._tau_inf = 1E12
            self._allow_stop = False
            # _silent_selection_of_traces is an attribute that defines if the
            # selection of traces should be add to record of actions.
            self._silent_selection_of_traces = False
            self._kmatrix_manual = False
            self._init_concentrations_manual = False
            self._last_params = None
            self._deconv = True
            self._exp_no = 1
            self.params = None
            self._weights = {'apply': False, 'vector': None, 'range': [],
                             'type': 'constant', 'value': 2}
            # _fit_number take record of global exponential and target fits ran.
            self._fit_number = 0
            self._params_initialized = False
            self._tau_inf = 1E12
            self._allow_stop = False
            self._model_params = None
            self.fit_records = UnvariableContainer(name="Fits")
            self._model_window = None
            self._readapting_params = False
            self._initialized()
            super().__init__(self.fit_records.global_fits, 
                             **self._experiment._units)

        def _initialized(self):
            """
            Finalize the initialization of the Experiment._Fit class
            """
            self.fit_records.global_fits = {}
            self.fit_records.single_fits = {}
            self.fit_records.bootstrap_record = {}
            self.fit_records.conf_interval = {}
            self.fit_records.target_models = {}
            self.fit_records.integral_band_fits = {}

        @property
        def _units(self):
            return self._experiment._units

        @_units.setter
        def _units(self, values):
            pass

        @property
        def _unit_formater(self):
            return self._experiment._unit_formater

        @_unit_formater.setter
        def _unit_formater(self, values):
            pass

        @property
        def time_unit(self):
            return f'{self._experiment._unit_formater._multiplicator.name}s'

        @time_unit.setter
        def time_unit(self, val: str):
            print("hola")
            try:
                print("hola")
                val = val.lower()
                self._experiment._unit_formater.multiplicator = val
                self._experiment._units['time_unit'] = self.time_unit
            except Exception:
                msg = 'An unknown time unit cannot be set'
                raise ExperimentException(msg)

        
        @property
        def wavelength_unit(self):
            return self._experiment._units['wavelength_unit']

        
        @wavelength_unit.setter
        def wavelength_unit(self, val: str):
            val = val.lower()
            if 'nanom' in val or 'wavelen' in val:
                val = 'nm'
            if 'centim' in val or 'wavenum' in val or 'cm' in val:
                val = 'cm-1'
            self._experiment._units['wavelength_unit'] = val

        @property
        def allow_stop(self):
            return self._allow_stop

        @allow_stop.setter
        def allow_stop(self, value):
            if type(value) == bool:
                self._allow_stop = value
            else:
                msg = "Type error, allow_stop should be a boolean"
                raise ExperimentException(msg)

        @property
        def fit_ready(self):
            if not self._params_initialized:
                if self._model_params is not None:
                    msg = "Target Model create. Initialize target parameters " \
                          "to perform a fit. Not ready to fit data yet"
                    return msg
                else:
                    return "Not ready to fit data"
            if self._params_initialized == "Model done":
                return "Target Model create, initialize target parameters;" \
                       "Not ready to fit data yet"
            else:
                msg = f"parameters for {self._params_initialized} " \
                      f"fit  with {self._exp_no} components, to " \
                      f"{self._experiment.selected_traces.shape[1]} traces"
                return msg

        @fit_ready.setter
        def fit_ready(self, value):
            print("fit_ready property cannot be set by the user")

        def get_weights(self):
            """
            Return the weights dictionary
            """
            return self._weights

        def print_fit_results(self, fit_number=None):
            """
            Print out a summarize result of a global fit.

            Parameters
            ----------
            fit_number: int or None (default None)
                defines the fit number of the results all_fit dictionary.
                If None the last fit in  will be considered.
            """
            if fit_number is None:
                fit_number = max(self._fits.keys())
            super().print_fit_results(fit_number=fit_number)
            if fit_number in self.fit_records.bootstrap_record.keys():
                print('\t The error has been calculated by bootstrap')
            if fit_number in self.fit_records.bootstrap_record.keys():
                print('\t The error has been calculated by an F-test')
            print('\n')

        def define_weights(self, rango, typo='constant', val=5):
            """
            Defines a an array that can be apply  in global fit functions as
            weights. The weights can be use to define areas where the minimizing
             functions is not reaching a good results, or to define areas that
             are more important than others in the fit. The fit with weights
            can be inspect as any other fit with the residual plot. A small
            constant value is generally enough to achieve the desire goal.

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
            self._weights = define_weights(self._experiment.x, rango, typo=typo,
                                           val=val)
            self._experiment._add_action("define weights")

        def initialize_exp_params(self, t0, fwhm, *taus, tau_inf=1E12,
                                  opt_fwhm=False, vary_t0=True,
                                  global_t0=True, y0=None):
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
                This constant modelled photoproducts formation with long
                decay times If None tau_inf is not added.
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
                corrected. Allows to fit the t0 globally (setting True), which
                is faster. In case this first option fit does not give good
                results in the short time scale the t0 can be independently
                setting to False which may give better results, if the
                correction of the chirp/GVD is not perfect (slower fit).

            y0: int or array (default None)
                Important: only applicable if fwhm is given.
                If given this value will fix the initial offset as a fix
                parameter. In case an array is given each trace will have a
                different value. In case an integer is pass all the traces will
                have this value fix
            """
            taus = list(taus)
            self._last_params = {'t0': t0, 'fwhm': fwhm, 'taus': taus,
                                 'tau_inf': tau_inf, 'opt_fwhm': opt_fwhm,
                                 'y0': y0, "vary_t0": vary_t0,
                                 "global_t0": global_t0}
            self._exp_no = len(taus)
            number_traces = self._experiment.selected_traces.shape[1]
            param_creator = GlobExpParameters(number_traces, taus)
            if fwhm is None:
                self._deconv = False
                vary_t0 = False
                correction = False
            else:
                gvd_corrected = self._experiment.preprocessing.GVD_corrected
                if global_t0 and not gvd_corrected:
                    correction = False
                elif not global_t0:
                    correction = False
                else:
                    correction = True
                self._deconv = True
                self._tau_inf = tau_inf
            param_creator.adjustParams(t0, vary_t0, fwhm, opt_fwhm,
                                       correction, tau_inf, y0)
            self.params = param_creator.params
            self._params_initialized = 'Exponential'
            if not self._readapting_params:
                self._experiment._add_action(f'new {self._params_initialized} '
                                             f'parameters initialized')

        def initialize_target_model_window(self):
            """
            Open a PyQT window where a Target model can be created.
            """
            if self._model_window is None:
                self._model_window = ModelWindow(
                    call_back=self._get_params_from_target_model_window)
            self._model_window.show()

        def _get_params_from_target_model_window(self):
            """callback function for self.model_window"""
            self._model_params = self._model_window.model.params
            key = len(self.fit_records.target_models) + 1
            model = copy.copy(self._model_window.model)
            self.fit_records.target_models[key] = model
            self._experiment._add_action(f'Target model from Window created')
            print("Initialize target params before running the fit")

        def initialize_target_model_manually(self, k_matrix: list,
                                             concentrations: list,
                                             names=None):
            """
            Build the k_matrix manually.

            Parameters
            ----------
            k_matrix: list of lists
                Contains all the information of k, rates. This parameter should
                be a list of list/tuples where every sub list should be of
                length and with the following form: [source, destination, rate
                constant, vary].
                if destination == source, parallel decay or terminal component.
                    i.e.: [1, 3, 1/0.7, True] --> component 1 gives component 3
                    with a rate constant of 1/0.7, and this parameter can be
                    optimized.

            concentrations: list
                a list containing the initial concentrations

            names: list (default None)
                A list containing the names of each specie. If given this will
                be used when reconstruct the model in the window model and in
                plot_concentration function. If not names will be given by
                increasing numbers.
            """
            param_creator = GlobalTargetParameters(1, None)
            param_creator.params_from_matrix(k_matrix=k_matrix,
                                             concentrations=concentrations)
            self._exp_no = param_creator.exp_no
            if names is None or len(names) != self._exp_no:
                names = [f"Species {i+1}" for i in range(self._exp_no)]
            param_creator.params.model_names = names
            self._model_params = param_creator.params
            self._experiment._add_action(f'Target model manually created')
            print("Initialize target params before running the fit")

        def initialize_target_params(self, t0, fwhm, model='auto',
                                     opt_fwhm=False, vary_t0=True,
                                     global_t0=True, y0=None):
            """
            function to initialize parameters for global fitting

            Parameters
            ----------
            t0: int or float
                the t0 for the fitting

            fwhm: float or None (default 0.12)
                FWHM of the the laser pulse use in the experiment
                If None. the deconvolution parameters will not be added

            model: auto or int
                if auto the initialize target model will be considered. If an
                integer is pass, the current model that will considered is
                stored in the fit_records.target_model dictionary, and should
                have been defined before.

            vary_t0: bool (default True)
                allows to optimize t0 when the sum of exponential is convolve
                with a gaussian. If there is no deconvolution t0 is always fix
                to the given value.

            opt_fwhm: bool (default False)
                allows to optimized the FWHM.
                Theoretically this should be measured externally and be fix
                (only applicable if fwhm is given)

            y0: int or float or list/1d-array (default None)
                If this parameter is pass y0 value will be a fixed parameter
                to the value passed. This affects fits with and without
                deconvolution. For a fit with deconvolution y0 is is added to
                negative delay offsets. For a fit without deconvolution y0 fit
                the offset of the exponential. If an array is pass this should
                have the length of the curves that want to be fitted, and for
                each curve the the y0 value would be different.
            """
            if type(model) == int:
                if model in self.fit_records.target_models.keys():
                    number_model = self.fit_records.target_models[model]
                    self._model_params = copy.copy(number_model.params)
                else:
                    msg = f"The model {model} is not in records"
                    raise ExperimentException(msg)
            self._last_params = {'t0': t0, 'fwhm': fwhm, 'taus': None,
                                 'tau_inf': None, 'opt_fwhm': opt_fwhm,
                                 'y0': y0, "vary_t0": vary_t0,
                                 "global_t0": global_t0}
            self._exp_no = self._model_params["exp_no"].value
            number_traces = self._experiment.selected_traces.shape[1]
            if fwhm is None:
                self._deconv = False
                vary_t0 = False
                correction = False
            else:
                self._deconv = True
                gvd_corrected = self._experiment.preprocessing.GVD_corrected
                if global_t0 and not gvd_corrected:
                    correction = False
                elif not global_t0:
                    correction = False
                else:
                    correction = True
            param_creator = GlobalTargetParameters(number_traces, None)
            param_creator.exp_no = self._exp_no
            param_creator.params = copy.copy(self._model_params)
            param_creator.adjustParams(t0, vary_t0=vary_t0,
                                       fwhm=fwhm,
                                       opt_fwhm=opt_fwhm,
                                       GVD_corrected=correction,
                                       y0=y0)
            self.params = param_creator.params
            self._params_initialized = "Target"
            if not self._readapting_params:
                self._experiment._add_action(f'new {self._params_initialized} '
                                             f'parameters initialized')

        """
        Fitting functions
        """
        def fit_global(self, vary=True, maxfev=5000, apply_weights=False,
                       use_jacobian=False, verbose=True):
            """
            Perform a exponential or a target global fit to the selected traces.
            The type of fits depends on the parameters initialized.

            Parameters
            ----------
            vary: bool or list of bool
                If True or False all taus are optimized or fixed.
                If a list, should be a list of bool equal with len equal to the
                number of  taus. Each element of the list defines if a initial
                taus should be optimized or not.

            maxfev: int (default 5000)
                maximum number of iterations of the fit.

            apply_weights: bool (default False)
                If True and weights have been defined, this will be applied in
                the fit (for defining weights) check the function define_weights.

            use_jacobian: bool (default False)
                If True the jacobian matrix is solved analytically. So far is
                only been implemented for exponential fit; is not available for
                Target fit

            verbose: bool (default True)
                If True, every 200 iterations the X2 will be printed out
            """
            gvd_corrected = self._experiment.preprocessing.GVD_corrected
            if hasattr(self._experiment.preprocessing.report, 'derivate_data'):
                derivative = True
            else:
                derivative = False
            if self._params_initialized == 'Exponential':
                minimizer = GlobalFitExponential(self._experiment.x,
                                                 self._experiment.selected_traces,
                                                 self._exp_no, self.params,
                                                 self._deconv, self._tau_inf,
                                                 GVD_corrected=gvd_corrected,
                                                 derivative=derivative)

            elif self._params_initialized == 'Target':
                minimizer = GlobalFitTarget(self._experiment.x,
                                            self._experiment.selected_traces,
                                            self._exp_no, self.params,
                                            deconv=self._deconv,
                                            GVD_corrected=gvd_corrected,
                                            derivative=derivative)
            else:
                msg = 'Parameters need to be initialized first'
                raise ExperimentException(msg)
            if apply_weights:
                minimizer.weights = self._weights
            if self.allow_stop:
                minimizer.allow_stop = True
            
            if self._params_initialized == 'Exponential':
                results = minimizer.global_fit(vary_taus=vary, maxfev=maxfev,
                                               apply_weights=apply_weights,
                                               use_jacobian=use_jacobian,
                                               verbose=verbose)
            else:
                # TODO add vary_k from vary need to modify Global fit
                results = minimizer.global_fit(maxfev=maxfev, # vary_k=vary,
                                               apply_weights=apply_weights,
                                               use_jacobian=use_jacobian,
                                               verbose=verbose)

                results.details['model_names'] = self._model_params.model_names
            # indicate if the fit is singular vectors
            results.details['svd_fit'] = self._experiment._SVD_fit
            # add selected wavelengths
            results.wavelength = self._experiment.selected_wavelength
            # add info on the average used in the selection of traces
            results.details['avg_traces'] = self._experiment._average_selected_traces
            # add the entire data set  and wavelengths in case the bootstrap is
            # done on the data and not the residues
            results.add_full_data_matrix(self._experiment.data,
                                         self._experiment.wavelength)

            self._fit_number += 1
            self.fit_records.global_fits[self._fit_number] = results
            msg = f'{self._params_initialized} fit performed'
            self._experiment._add_action(msg)
            self._update_last_params(results.params)

        def _update_last_params(self, params):
            """
            Function updating parameters after a global fit
            """
            if self._params_initialized == 'Exponential':
                self._last_params['t0'] = params['t0_1'].value
                self._last_params['taus'] = [params['tau%i_1' % (i + 1)].value
                                             for i in range(self._exp_no)]
            elif self._params_initialized == 'Target':
                self._last_params['t0'] = params['t0_1'].value
                for i in range(self._exp_no):
                    for ii in range(self._exp_no):
                        self._model_params["k_%i%i" % (i+1, ii+1)].value = \
                            self.params["k_%i%i" % (i+1, ii+1)].value
            else:
                pass

        def fit_single_exp(self, wave: int, average: int, t0: float, fwhm: float,
                           *taus, vary=True, tau_inf=1E12, maxfev=5000,
                           apply_weights=False, opt_fwhm=False, plot=True):
            """
            Perform an exponential fit to a single trace

            Parameters
            ----------
            wave: int or float
                the closest value in the wavelength vector traces will be
                selected.

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
                If True or False all taus are optimized or fixed. If a list,
                should be a list of bool equal with len equal to the number of
                taus. Each entry defines if a initial taus should be optimized
                or not.

            maxfev: int (default 5000)
                maximum number of iterations of the fit.

            apply_weights: bool (default False)
                If True and weights have been defined, this will be applied in
                the fit (for defining weights) check the function define_weights.

            plot: bool (default True)
                If True the results are automatically plotted and a figure and
                axes are return.
            """
            taus = list(taus)
            trace, wave = select_traces(self._experiment.data,
                                        self._experiment.wavelength,
                                        [wave], average)
            results = self._one_trace_fit(trace, t0, fwhm, *taus, vary=vary,
                                          tau_inf=tau_inf, maxfev=maxfev,
                                          apply_weights=apply_weights,
                                          opt_fwhm=opt_fwhm)
            results.wavelength = wave
            key = len(self.fit_records.single_fits) + 1
            self.fit_records.single_fits[key] = results
            self._experiment._add_action('Exponential single fit performed')
            if plot:
                fig, ax = self.plot_single_fit(key)
                return fig, ax

        def fit_integral_band_exp(self, wave_range: list, t0: float,
                                  fwhm: float, *taus, vary=True, tau_inf=1E12,
                                  maxfev=5000, apply_weights=False,
                                  opt_fwhm=False, plot=True):
            """
            Perform an exponential fit to an integrated are of the spectral
            range of the data set. This type of fits allows for example to
            identify time constants attributed to cooling since the integration
            compensate the effects and the contribution of this type of
            phenomena to the decay decreases or disappears.

            Parameters
            ----------
            wave_range: list (length 2) or float
                The area between the two entries of the wavelength range is
                integrated and fitted.

            t0: int or float
                the t0 for the fitting

            fwhm: float or None
                FWHM of the the laser pulse use in the experiment
                If None. the deconvolution parameters will not be added

            taus: int or float
                initial estimations of the decay times

            tau_inf: int or float (default 1E12)
                allows to add a constant decay value to the parameters.
                This modelled photo-products formation with long decay times
                If None tau_inf is not added.
                (only applicable if fwhm is given)

            opt_fwhm: bool (default False)
                allows to optimized the FWHM.
                Theoretically this should be measured externally and be fix
                (only applicable if fwhm is given)

            vary: bool or list of bool
                If True or False all taus are optimized or fixed. If a list,
                should be a list of bool equal with len equal to the number of
                taus. Each entry defines if a initial taus should be
                optimized or not.

            maxfev: int (default 5000)
                maximum number of iterations of the fit.

            apply_weights: bool (default False)
                If True and weights have been defined, this will be applied in
                the fit (for defining weights) check the function define_weights

            plot: bool (default True)
                If True the results are automatically plotted and a figure and
                axes are return.
            """
            taus = list(taus)
            indexes = [np.argmin(abs(self._experiment.wavelength-wave_range[0])),
                       np.argmin(abs(self._experiment.wavelength-wave_range[1]))]

            x_to_integrate = self._experiment.wavelength[indexes[0]:indexes[1]]
            trace = np.array([np.trapz(self._experiment.data[i, indexes[0]:indexes[1]],
                                       x=x_to_integrate)
                              for i in range(len(self._experiment.data))])

            trace = trace.reshape((trace.shape[0], 1))
            results = self._one_trace_fit(trace, t0, fwhm, *taus, vary=vary,
                                          tau_inf=tau_inf, maxfev=maxfev,
                                          apply_weights=apply_weights,
                                          opt_fwhm=opt_fwhm)
            results.details['integral band'] = wave_range
            key = len(self.fit_records.integral_band_fits) + 1
            self.fit_records.integral_band_fits[key] = results

            msg = f'Integral band fit between {wave_range[0]} and ' \
                  f'{wave_range[1]} performed'
            self._experiment._add_action(msg)
            if plot:
                fig, ax = self.plot_integral_band_fit(key)
                return fig, ax

        def _one_trace_fit(self, trace, t0, fwhm, *taus,
                           vary=True, tau_inf=1E12,
                           maxfev=5000, apply_weights=False,
                           opt_fwhm=False, y0=None):
            """
            Real fitting function used by "integral_band_exp_fit"
            and "single_exp_fit"
            """
            # print(taus)
            param_creator = GlobExpParameters(1, list(taus))
            chirp_corrected = self._experiment.preprocessing.GVD_corrected
            param_creator.adjustParams(t0, vary, fwhm, opt_fwhm,
                                       chirp_corrected,
                                       tau_inf, y0)
            # print(param_creator.params)
            deconv = True if fwhm is not None else False

            minimizer = GlobalFitExponential(self._experiment.x,
                                             trace,
                                             len(taus),
                                             params=param_creator.params,
                                             deconv=deconv, tau_inf=tau_inf,
                                             GVD_corrected=False)
            results = minimizer.global_fit(vary, maxfev, apply_weights)
            return results

        def plot_single_fit(self, fit_number=None, details=True):
            """
            Function that generates a figure with the results of the fit stored
             in the single_fits

            Parameters
            ----------
            fit_number: int or None (default None)
                defines the fit number of the results single_fits dictionary. If
                None the last fit in  will e considered

            details: bool (default True)
                If True the decay times obtained in the fit are included in the
                figure
                
            """
            if fit_number is None:
                fit_number = len(self.fit_records.single_fits)
            if fit_number in self.fit_records.single_fits.keys():
                return self._plot_single_trace_fit(self.fit_records.single_fits,
                                                   fit_number, details)
            else:
                if fit_number == 0:
                    msg = "So far no single fits are perfomred"
                else:
                    msg = 'Fit number not in records'
                raise ExperimentException(msg)

        def plot_integral_band_fit(self, fit_number=None, details=True):
            """
            Function that generates a figure with the results of the fit stored
            in the integral_band_fits

            Parameters
            ----------
            fit_number: int or None (default None)
                defines the fit number of the results integral_band_fits
                dictionary.
                If None the last fit in  will be considered

            details: bool (default True)
                If True the decay times obtained in the fit are included in the
                figure
            """
            if fit_number is None:
                fit_number = len(self.fit_records.integral_band_fits)

            if fit_number in self.fit_records.integral_band_fits.keys():

                result = self.fit_records.integral_band_fits
                fig, ax = self._plot_single_trace_fit(result,
                                                      fit_number,
                                                      details)

                rang = self.fit_records.integral_band_fits[fit_number].details['integral band']

                if self._units['wavelength_unit'] == 'cm-1':
                    w_unit = 'cm$^{-1}$'
                else:
                    w_unit = self._units['wavelength_unit']
                ax[1].legend(['_', f'Integral band {rang[0]}-{rang[1]} {w_unit}'])
                return fig, ax
            else:
                if fit_number == 0:
                    msg = "So far no single fits are perfomred"
                else:
                    msg = 'Fit number not in records'
                raise ExperimentException(msg)

        def _plot_single_trace_fit(self, container, fit_number,
                                   add_details=True):
            """
            Base plot function used by "plot_integral_band_fit"
            and "plot_single_fit"
            """
            plotter = ExploreResults(container[fit_number], **self._units)
            values = plotter._get_values(fit_number=1)
            data = values[1]
            params = values[3]
            exp_no = values[4]
            deconv = values[5]
            tau_inf = values[6]
            fig, ax = plotter.plot_global_fit()

            if add_details:
                if data[0] <= 0:
                    anchor_location = "lower center"
                else:
                    anchor_location = "upper center"
                text = plotter._legend_plot_DAS(params, exp_no, deconv,
                                                tau_inf, 'Exponential', 2)
                # next if statement is rot remove the 'offset' word from the
                # text only in case there was no deconvolution
                if not deconv:
                    text = text[:-1]
                textstr = '\n'.join(text)
                texto = AnchoredText(s=textstr, loc=anchor_location)
                ax[1].add_artist(texto)

            return fig, ax

        def fit_with_SVD_fit_result(self, fit_number=None, fit_data='all'):
            # TODO finish function
            """
            Not finished
            """
            # _get_values is a heritage function from ExploreResults class
            values = self._get_values(fit_number=1)
            x = values[0]
            data = values[1]
            wavelength = values[2]
            params = values[3]
            exp_no = values[4]
            deconv = values[5]
            tau_inf = values[6]
            svd_fit = values[7]
            type_fit = values[8]
            derivative_space = values[6]

            if fit_data == 'all':
                data_fit = self._experiment.selected_traces
            elif fit_data == 'selected':
                data_fit = self._experiment.data
            else:
                msg = "fit_data should be 'all' or 'selected'"
                raise ExperimentException(msg)

            if type_fit == 'Exponential':
                taus = [params['tau%i_1' % (i+1)].value for i in range(exp_no)]
                t0 = params['t0_1'].value
                if deconv:
                    fwhm = params['fwhm_1'].value
                else:
                    fwhm = None
                param_creator = GlobExpParameters(data_fit.shape[1], taus)
                gvd_correct = self._experiment.preporcessing.GVD_corrected
                param_creator.adjustParams(t0, True, fwhm, False,
                                           gvd_correct,
                                           tau_inf)

                params_fit = param_creator.params
                minimizer = GlobalFitExponential(self._experiment.x,
                                                 data_fit,
                                                 exp_no, params_fit,
                                                 deconv, tau_inf, False)
                minimizer.pre_fit()
            else:
                # Todo

                pass

        def _readapt_params(self):
            """
            Function to automatically re-adapt parameters to a new selection of
            traces from the original data set.
            """
            self._readapting_params = True  # avoid to add_action initialize_params
            if type(self._params_initialized) != bool:
                t0 = self._last_params['t0']
                fwhm = self._last_params['fwhm']
                opt_fwhm = self._last_params['opt_fwhm']
                vary_t0 = self._last_params["vary_t0"]
                global_t0 = self._last_params["global_t0"]
                y0 = self._last_params['y0']
                if self._params_initialized == 'Exponential':
                    previous_taus = self._last_params['taus']
                    tau_inf = self._last_params['tau_inf']
                    self.initialize_exp_params(t0, fwhm, *previous_taus,
                                               tau_inf=tau_inf,
                                               opt_fwhm=opt_fwhm,
                                               vary_t0=vary_t0,
                                               global_t0=global_t0,
                                               y0=y0)
                elif self._params_initialized == 'Target':
                    # previous_model = self._last_params['model_params']
                    # self._model_params = previous_model
                    fwhm = self._last_params['fwhm']
    
                    self.initialize_target_params(t0, fwhm, model='auto',
                                                  opt_fwhm=opt_fwhm,
                                                  vary_t0=vary_t0,
                                                  global_t0=global_t0, 
                                                  y0=y0)

            else:
                # if parameters are not initialize pass
                pass
            self._readapting_params = False

        def fit_bootstrap(self, fit_number: int, boots: int, size=25,
                          data_from="residues", workers=2):
            """
            Perform a bootstrap fit. It can be done to either the residues or to
            the data.

            Parameters
            ----------
            fit_number: int
                Defines the fit number  that will be consider to perform the
                bootstrap. Is the key of the results all_fit dictionary.

            boots: int
                Defines the number of data sets that will be generated from
                either the residues or the original data.

            size: int (default 25)
                Only important if the data_from is residues, defines the
                percentage of residues that will be shuffle.
                can be 10, 15, 20, 25, 33 or 50. We recommend to uses 25 or 33.

            data_from: valid "residues"; "fitted_data"; "full_data_matrix"
                If "residues" data are simulated shuffling residues with the
                model.

                If "fitted_data" data are simulated from a random selection of
                the  original fitted traces with replacement.

                If "full_data_matrix" data are simulated from a random selection
                of the original entire full data matrix with replacement.

                We recommend to either use 'residues' or 'full_data_matrix'.

            workers: int (default 2)
                If workers > 1, then the calculation will be run in parallel in
                different CPU cores.Workers define the number of CPU cores that
                will be used. We recommend to used as maximum half of the CPU
                cores, and up to 4 if the analysis is run in a regular computer.

            """
            can_run = self._assert_boot_strap_can_run(fit_number)
            if not can_run:
                msg = "Please perform a global fit first"
                raise ExperimentException(msg)

            previous_results = self._bootstrap_previous_results(fit_number)
            fit_results = self.fit_records.global_fits[fit_number]

            if workers >= 2:
                parallel_compute = True
            elif workers == 1:
                parallel_compute = False
            else:
                workers == 1
                parallel_compute = False

            boot_strap = BootStrap(fit_results,
                                   previous_results,
                                   workers, self._experiment.time_unit)

            boot_strap.generate_datasets(boots, data_from=data_from, size=size)
            boot_strap.fit_bootstrap(parallel_computing=parallel_compute)
            self.fit_records.bootstrap_record[fit_number] = \
                boot_strap.bootstrap_result
            conf = boot_strap.confidence_interval
            self.fit_records.conf_interval[fit_number] = conf

        def plot_bootstrap_result(self, fit_number,  param_1,
                                  param_2=None, kde=False):
            """
            Plot the bootstrap histogram of the decay times calculated
            If param_1 and param_2 are given a correlation plot with the
            histogram distributions is plot. If a single param is given only the
            histogram distribution is plot.

            Parameters
            ----------
            fit_number: int
                Defines the fit number  that will be consider to perform the
                bootstrap. Is the key of the results all_fit dictionary

            param_1: str or int
               name of the tau to be plotted;
                i.e.: for first decay time --> if string: tau1, if integer: 1

            param_2: str or int or None
                name of the tau to be plotted;
                i.e.: for third decay time --> if string: tau3, if integer: 3

            kde: bool (default True)
                Defines if the kernel density estimation is plotted.
            """
            previous_results = self._bootstrap_previous_results(fit_number)
            fit_results = self.fit_records.global_fits[fit_number]
            boot_strap = BootStrap(fit_results,
                                   previous_results,
                                   1, self._experiment.time_unit)
            return boot_strap.plotBootStrapResults(param_1, param_2, kde)

        def get_result(self, fit_number=None, fit_type="global"):
            """
            Returns the result of a fit performed. Each type_fit has
            independent fit_number

            Parameters
            ----------
            fit_number: int or None
                The fit number that will return

            fit_type: valid "global", "single", "integral", "bootstrap"
                Defines the fit that will be return if has been performed
                For "global", "single" and "integral" an GlobalFitResult will be
                return. This object can be pass to ExploreResults class.
                For "bootstrap" a pandasDataFrame is return
            """

            if fit_type == "global":
                container = self.fit_records.global_fits
            elif fit_type == "single":
                container = self.fit_records.integral_band_fits
            elif fit_type == "integral":
                container = self.fit_records.single_fits
            elif fit_type == "bootstrap":
                container = self.fit_records.bootstrap_record
            if fit_number is None:
                fit_number = max(self._fits.keys())
            if fit_number in container.keys():
                return container[fit_number]
            else:
                msg = f"Fit number {fit_number}, not in {fit_type} fit records"
                raise ExperimentException(msg)

        def _bootstrap_previous_results(self, fit_number):
            """
            Return the results of the bootstrap for a given fit_number
            if there are any
            """
            if fit_number in self.fit_records.bootstrap_record.keys():
                results = self.fit_records.bootstrap_record[fit_number]
            else:
                results = None
            return results

        def _assert_boot_strap_can_run(self, fit_number: int) -> bool:
            if fit_number in self.fit_records.global_fits.keys():
                can_run = True
            else:
                can_run = False
            return can_run
    """
    Data selection and restoration functions
    """
    def select_traces(self, points=10, average=1, avoid_regions=None):
        """
        Method to select traces from the data, the selected traces are stored
        in the selected_traces attribute. If the parameters have been
        initialize automatically re-adapts them to the new selected traces.
        (The function assumes that the wavelength vector is sorted from low
        to high values)

        Parameters
        ----------
        points: int or list or "auto" (default 10)
            If type(space) =int: a series of traces separated by the value
            indicated will be selected.
            If type(space) = list: the traces in the list will be selected.
            If space = auto, the number of returned traces is 10 and equally
            spaced along the wavelength vector and average is set to 0

        average: int (default 1)
            Binning points surrounding the selected wavelengths.
            e. g.: if point is 1 trace = mean(index-1, index, index+1)

        avoid_regions: list of list (default None)
            Defines wavelength regions that are avoided in the selection when
            space is an integer. The sub_list should have two elements defining
            the region to avoid in wavelength values

            i. e.: [[380,450],[520,530] traces with wavelength values between
                    380-450 and 520-530 will not be selected
        """
        super().select_traces(points, average, avoid_regions)
        self.fitting._readapt_params()
        self._average_selected_traces = average if points != 'all' else 0
        if self._silent_selection_of_traces:
            self._silent_selection_of_traces = False
        else:
            self._add_action("Selected traces")

    def select_region(self, mini, maxi):
        """
        Select a region of the data as selected traces according to the closest
        values of mini and maxi to the wavelength vector. If the parameters have
        been initialize automatically re-adapts them to the new selected traces
        (The function assumes wavelength vector is sorted from low to
        high values)

        Parameters
        ----------
        mini: int, float or None
          data higher than this value is kept

        maxi: int, float or None
          data lower than this value is kept
        """
        new_data, new_wave = Prep.cut_columns(self.data,
                                              self.wavelength,
                                              mini, maxi, True)
        self.selected_traces, self.selected_wavelength = new_data, new_wave
        self.fitting._readapt_params()
        self._average_selected_traces = 0
        self._add_action("Selected region as traces")

    """
    Other private methods
    """
    def _add_action(self, value, re_select_traces=False):
        """
        add action to action records
        """
        val = len(self._action_records.__dict__) - 2
        self._action_records.__setattr__(f"_{val}", value)
        if re_select_traces:
            self._re_select_traces()

    def _re_select_traces(self):
        self._silent_selection_of_traces = True
        if self._SVD_fit:
            self._calculateSVD()
            self.select_SVD_vectors(self.selected_traces.shape[1])
        else:
            avg = self._average_selected_traces
            wave = list(self.selected_wavelength)
            self.select_traces(wave, avg)
            # val = len(self.action_records.__dict__) - 3
            # delattr(self.action_records, f"_{val}" )
