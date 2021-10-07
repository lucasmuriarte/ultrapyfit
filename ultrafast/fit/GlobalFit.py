# -*- coding: utf-8 -*-
"""
Created on Sat Mars 13 14:35:39 2021
@author: Lucas
"""
import numpy as np
import lmfit
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.fit.GlobalParams import GlobExpParameters
from ultrafast.utils.divers import UnvariableContainer, solve_kmatrix
import copy
import pickle
from ultrafast.graphics.ExploreResults import ExploreResults
import matplotlib.pyplot as plt
import ctypes
import threading


# define a thread which takes input
class InputThread(threading.Thread):
    def __init__(self, function):
        super(InputThread, self).__init__()
        self.daemon = True
        self.last_user_input = None
        self.function = function
        
    def run(self):
        while self._stop:
            self.last_user_input = input('Type "stop" to stop fit:  ')
            if self.last_user_input == 'stop':
                self.function()
                self.stop()
                break

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def stop(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(
                                                             SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')


class Container:
    """
    Object where once an attribute has been set cannot be modified if
    self.__frozen = True
    """
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)


class GlobalFitResult:
    """
    Contain all attributes that an lmfit minimizer result has, and add some
    specific details of the fit to this results. These are:
    x: 1darray
        x-vector, normally time vector

    data: 2darray
        Array containing the data, the number of rows should be equal to
        the len(x)
    wavelength: 1darray
            wavelength vector
    details:
        A dictionary containing specific details of the fit performed
    """
    def __init__(self, result):
        for key in result.__dict__.keys():
            setattr(self, key, result.__dict__[key])
        self.details = None

    def add_data_details(self, data, details):
        """
        Add as attributes to the lmfit results object: these are the data, the
        time the wavelength. Also add fit details such as the number of
        exponential if convolve with a gaussian or not, tau_inf, maxfev and
        other properties that are later use by UltrafastExperiments class and
        other classes as ExploreResults.
        """
        self.x = data.x
        self.data = data.data
        self.wavelength = data.wavelength
        self.details = details
        self.details['time_constraint'] = False
        
    def das_to_sas(self):
        """
        Recalculates DAS spectra into EAS, treating y_inf as final product.
        Replaces DAS with EAS, so one should treat this data as EAS after
        calling this method.
        It works with matrixes and should be correct for any number of exp's.
        """
        
        params = self.params.copy()
        no_of_wavelengths = self.wavelength.shape[0]
        tau_inf_enabled = True #set here if it is enabled
        try:
            y_inf = params['yinf_' + str(1)]
        except:
            tau_inf_enabled = False
        no_of_exps = self.exp_no #load here number of exponentials
        
        ##########
        if(tau_inf_enabled):
            size_of_kmatrix = no_of_exps + 1
        else:
            size_of_kmatrix = no_of_exps

        #let's build kmatrix of the sequential model        
        kmatrix = np.zeros((size_of_kmatrix,size_of_kmatrix))
        k_values = []
        for i in range(no_of_exps):
            k_value = 1/params['tau%i_' % (i+1) + str(1)].value
            kmatrix[i,i] = -k_value
            if(i+1 < size_of_kmatrix):
                kmatrix[i+1,i] = k_value
            k_values.append(k_value)
                
        #initialize initial values, since it is sequential, only first is 1
        c_initials = np.zeros(size_of_kmatrix)
        c_initials[0] = 1.0

        #now extract eigenvalues and eigenvectors, to be able to reduce kmatrix
        #to the only-diagonal form. then we can easily differentiate and get solution
        eigs_out, vects_out = np.linalg.eig(kmatrix)
        
        #note that eigenvalues can come out not ordered. we want them in the
        #same order and values as k1,k2,k3... note that the same values are
        #obtained only if kmatrix describes sequential model. more complicated
        #models can give different k values than these used to build kmatrix              
        if(tau_inf_enabled):
            oryginal_k_order=-np.array(k_values+[0.0])
        else:
            oryginal_k_order=-np.array(k_values)     
        ks_ordering = np.argsort(oryginal_k_order)
        ks_reverse_ordering = np.argsort(ks_ordering)
        
        #sort eigenthings
        sort_ordering = np.argsort(eigs_out)
        eigs_sorted = eigs_out[sort_ordering]
        vects_sorted = vects_out[:,sort_ordering]
        
        #order eigenthings like k1,k2,k3....
        eigs = eigs_sorted[ks_reverse_ordering]
        vects = vects_sorted[:,ks_reverse_ordering]
        
        #then solve linear equation, where t=0 so you have
        #eigvects_matrix*vect_of_concentrations = vect_of_initial_values
        #by this you get coeffs which are before diagonalized exp functions
        #it is crucial, to have 1 before exp which is first in cascade, this
        #is why we sorted things before
        coeffs = np.linalg.solve(vects, c_initials)
        
        #ok, now you make diagonal array with these coeffs
        em_matrix = np.identity(coeffs.shape[0]) * np.transpose(coeffs[np.newaxis])
        #and you multiply eigenvector matrix by this. so you have:
        #fit_of_data = eas_array * d_matrix * exp_matrix
        #fit_of_data = das_array * exp_matrix
        d_matrix = np.dot(vects,em_matrix)
        
        #now you can just transpose that, and by comparison of das and sas,
        #you can get eas values from linear equation:
        d_t = np.transpose(d_matrix)
        #like below, but need to do this in loop for all kinetics:
        #EASv = np.linalg.solve(d_t, DASv)  
            
        #idea is to iterate slowly over every kinetic, and change exp-associated
        #preexp factors into species associated preexp factors
        for wavelength_num in range(no_of_wavelengths):
            DASv = [self.params['pre_exp%i_' % (i+1) + str(wavelength_num+1)].value 
                                for i in range(no_of_exps)]
            if(tau_inf_enabled):
                DASv.append(self.params['yinf_' + str(wavelength_num+1)].value)
            
            EASv = np.linalg.solve(d_t, np.array(DASv)) 
            
            #lets replace das with eas:
            for i in range(no_of_exps):
                params['pre_exp%i_' % (i+1) + str(wavelength_num+1)].set(value=EASv[i])
            if(tau_inf_enabled):    
                params['yinf_' + str(wavelength_num+1)].set(value=EASv[no_of_exps])
            

        #lets put back the params, but now pre_exps are EAS, not DAS!
        self.params = params
        
    def das_to_sas_3exp(self):
        """
        Recalculates DAS spectra into EAS, treating y_inf as final product.
        Replaces DAS with EAS, so one should treat this data as EAS after
        calling this method.
        This is special case where we have 3 exponentials + offset.
        It is calculated manually, to check if it generates the same result as 
        das_to_sas method (only for 3exp+offset). 
        Compare results of this func with result of das_to_sas() to be sure that
        everything is correct.
        """        

        params = self.params.copy()
        no_of_wavelengths = self.wavelength.shape[0]
        tau_inf_enabled = True #set here if it is enabled
        try:
            y_inf = params['yinf_' + str(1)]
        except:
            tau_inf_enabled = False
        no_of_exps = self.exp_no #load here number of exponentials
        
        if(no_of_exps != 3):
            raise Exception("This conversion method cannot be used in this case!")
        if(tau_inf_enabled is not True):
            raise Exception("This conversion method cannot be used in this case!")

        k1 = 1/params['tau%i_' % (1) + str(1)].value
        k2 = 1/params['tau%i_' % (2) + str(1)].value
        k3 = 1/params['tau%i_' % (3) + str(1)].value

        for wavelength_num in range(no_of_wavelengths):
            DAS1 = self.params['pre_exp%i_' % (1) + str(wavelength_num+1)].value
            DAS2 = self.params['pre_exp%i_' % (2) + str(wavelength_num+1)].value
            DAS3 = self.params['pre_exp%i_' % (3) + str(wavelength_num+1)].value
            offset = self.params['yinf_' + str(wavelength_num+1)].value

            EAS1 = DAS1 + DAS2 + DAS3 + offset*(-k1/(k2-k1)-k1*k2/((k2-k1)*(k3-k1))+k2*k3/((k2-k1)*(k3-k1)))
            EAS2 = DAS2*((k1-k2)/k1) + DAS3*((k1-k3)/k1) + offset
            EAS3 = DAS3*((k1-k3)*(k2-k3)/(k1*k2)) + offset
            #offset = EAS4 = DAS4 = offset

            #lets replace das with eas:
            params['pre_exp%i_' % (1) + str(wavelength_num+1)].set(value=EAS1)
            params['pre_exp%i_' % (2) + str(wavelength_num+1)].set(value=EAS2)
            params['pre_exp%i_' % (3) + str(wavelength_num+1)].set(value=EAS3)


        #lets put back the params, but now pre_exps are EAS, not DAS!
        self.params = params


    def save(self, name):
        path = name + '.res'
        with open(path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    # static method intentionally left without the @staticmethod decorator
    def load(filename):
        with open(filename, "rb") as f:
            loaded_results = pickle.load(f)
        return loaded_results


class GlobalFit(lmfit.Minimizer, ModelCreator):
    def __init__(self,
                 x,
                 data,
                 exp_no,
                 params,
                 deconv=True,
                 tau_inf=1E+12,
                 GVD_corrected=True,
                 wavelength=None,
                 **kwargs):
        weights = dict({'apply': False, 'vector': None, 'range': [],
                        'type': 'constant', 'value': 2,
                        'derivative': False}, **kwargs)
        self._derivative = weights.pop('derivative')
        self.weights = weights
        self.x = x
        self.data = data
        if wavelength is not None:
            self.wavelength = wavelength
        else:
            self.wavelength = np.array([i for i in
                                        range(1, self.data.shape[1] + 1)])
        self.params = params
        self._capture_params = []
        self.deconv = deconv
        self.tau_inf = tau_inf
        self.exp_no = exp_no
        self.GVD_corrected = GVD_corrected
        self.fit_type = None
        self._number_it = 0
        self._stop_manually = False
        self._prefit_done = False
        self._data_ensemble = UnvariableContainer(x=x, data=data,
                                                  wavelength=self.wavelength)
        # self._progress_result = None
        # self._allow_stop = False
        self._allow_stop = False
        self.thread = None
        self.fit_completed = False
        self._ax = None
        self._fig = None
        ModelCreator.__init__(self, self.exp_no, self.x, self.tau_inf)
        lmfit.Minimizer.__init__(self, self._objective, params,
                                 nan_policy='propagate',
                                 iter_cb=self.print_progress)

    def minimize(self, method='leastsq', params=None, max_nfev=None, **kws):
        """
        Modified minimize function to output GloablFitResult instead of
        lmfit results, except for this works identically as lmfit minimize.
        Check lmfit minimize for more information
        """
        result = super().minimize(method=method, params=params,
                                  max_nfev=max_nfev, **kws)
        result = GlobalFitResult(result)
        details = self._get_fit_details()
        details['maxfev'] = max_nfev
        result.add_data_details(self._data_ensemble, details)
        if not self.weights['apply']:
            result.weights = False
        else:
            result.weights = self.weights
        return result
    
    @property
    def allow_stop(self):
        """
        controlles if is posible to stop the fit or not
        """
        return self._allow_stop
    
    @allow_stop.setter
    def allow_stop(self, value: bool):
        # if value:
        #     msg = 'WARNING: Setting this value to True, ' \
        #           'might generate "Bad address" error'
        #     print(msg)
        self._allow_stop = value
    
    def pre_fit(self):
        """
        Intended to be a method that optimized the pre_exponential factors
        trace by trace without optimizing the decay times (taus).
        """
        pass

    def _objective(self, params):
        """
        The function that should be minimize.It should output an array of
        residues, generated by data-model. The array of residues should be
        flatten. Check lmfit for more details.
        """
        pass

    def global_fit(self, maxfev=None, apply_weights=False, method='leastsq', 
                   **kws):
        """
        Method to fit the data to a model. Returns a modified lmfit result
        object.

        Parameters
        ----------
        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        apply_weights: bool (default False)
            If True and weights have been defined, this will be applied in the
            fit (for defining weights) check the function define_weights.

        method: default (leastsq)
            Any valid method that can be used by lmfit minimize function

        kws:
            Any valid kwarg that can be used by lmfit minimize function
        """
        # if plot:
        #     self._plot = True
        self.fit_completed = False
        if not self._prefit_done:
            print('Preparing Fit')
            self.pre_fit()
            print('Starting Fit')
        if apply_weights and len(self.weights['vector']) == len(self.x):
            self.weights['apply'] = True
        else:
            self.weights['apply'] = False
        self.fit_completed = False
        if self._allow_stop:
            user_stop = InputThread(self.stop_fit)
            user_stop.start()
        if maxfev is not None:
           maxfev = int(maxfev)
        resultados = self.minimize(method=method, params=self.params, 
                                   max_nfev=maxfev, **kws)
        if self._allow_stop:
            user_stop.stop()
            user_stop.join()
        self.params = copy.copy(resultados.params)
        self._number_it = 0
        self.fit_completed = True
        if self._stop_manually:
            print('Fit stop manually')
            self._abort = False
            self._stop_manually = False
        if self.weights['apply']:
            self.weights['apply'] = False
        return resultados

    def _single_fit(self, params, function, i, extra_params=None):
        """
        Generate a single residue for one trace (used by prefit)
        """
        if extra_params is None:
            model = function(params, i)
        else:
            model = function(params, i, extra_params)
        if type(self.deconv) == bool:
            if self.deconv:
                data = self.data[:, i] - model
            else:
                t0 = params['t0_%i' % (i + 1)].value
                index = np.argmin([abs(i - t0) for i in self.x])
                data = self.data[index:, i] - model
        else:
            data = self.data[:, i] - model
        return data

    def _generate_residues(self, function, params, extra_param=None):
        """
        Generate a single residue for one trace (used by global_fit)
        """
        ndata, nx = self.data.shape
        data = self.data[:]
        resid = data * 1.0
        for i in range(nx):
            if extra_param is not None:
                resid[:, i] = data[:, i] - function(params, i, extra_param)
            else:
                resid[:, i] = data[:, i] - function(params, i)
            if self.weights['apply']:
                resid[:, i] = resid[:, i] * self.weights['vector']
        return resid

    def _get_fit_details(self):
        """
        return details of the object
        """
        if type(self.deconv) == bool:
            deconv = self.deconv
        else:
            deconv = False
        tau_inf = self.tau_inf if deconv else None
        details = {'exp_no': self.exp_no,
                   'deconv': self.deconv,
                   'type': self.fit_type,
                   'tau_inf': tau_inf,
                   'svd_fit': False,
                   'derivative': self._derivative,
                   'avg_traces': 'unknown'}
        return details

    def print_progress(self, params, iter, resid):
        """
        call back function of the minimizer, it can be used to stop the fit.
        If not print the number of iterations and the chi square every 200
        iterations.
        """
        get_stop = self._stop_manually
        if self._number_it % 200 == 0:
            print("Iteration: " + str(self._number_it) + ", chi2: " +
                  str(sum(np.abs(resid.flatten()))))
        if get_stop:
            return True
            # self._update_progress_result(params)
            # if self._plot:
            #     self.plot_progress()
    
    def stop_fit(self):
        """
        use to stop the fit
        """
        self._stop_manually = True


class GlobalFitExponential(GlobalFit):
    """
    Class that does a global fit using exponential models. This class is uses by
    the function, globalfit_gauss_exponential and globalfit_exponential
    functions. A global fit evaluate the times from all the traces (taus are
    globally fitted), while the pre_exponential values (pre_exp) are estimated
    independently from each trace. The pre_exp values give later the decay
    associated spectra (DAS). The Class do not generates the parameters
    automatically.

    Attributes
    ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            array containing the data, the number of rows should be equal to the
            len(x)

        exp_no: int
            number of exponential that will be used to fit the data.

        params: lmfit parameters object
            object containing the initial parameters values used to build an
            exponential model. These parameters are iteratively optimize to
            reduce the residual matrix formed by data-model (error matrix)
            using Levenberg-Marquardt algorithm.

        deconv: bool
            If True the fitting functions will search for the deconvolution
            parameter ("fwhm") in the params attribute, and the the model is a
            weighted sum of Gauss modified exponential functions. If False the
            the model is a weighted sum of exponential functions, and params
            should not contain the fwhm entrance.

        tau_inf: float or int or None
            An extra decay time use to evaluate possible photoproduct. This
            should be used if the signal at long delay times is not completely
            recovered and if deconv is set to True. If the signal at long delay
            times is zero tau_inf should be set to None.
            (only affects if deconv is True)

        GVD_corrected: bool
            Defines if the chrip or group velocity dispersion (GVD) has been
            corrected. If True t0 is globally optimized (faster). If False t0 is
            separately optimized for each trace (slower). Notice in some cases
            even if the chirp or GVD has been corrected, very small variations
            of t0 that might be imperceptible and small may generate problems in
            the fit, setting this parameter to False can help to acquire
            overcome this problem although the fit will take longer.
            (only affects if deconv is True)

        weights: dictionary
            This dictionary controls if the fitting weights are applied,
            the keys are:
            "apply": Bool
            "vector": weighting vector.
            'type': type of vector defined in the ,
            'range': time range according to x-vector of the weights that are
                     different than 1
            'value': val weighting value

            The dictionary keys can be passes as kwargs when instantiating the
            object
    """

    def __init__(self,
                 x,
                 data,
                 exp_no,
                 params,
                 deconv=True,
                 tau_inf=1E+12,
                 GVD_corrected=True,
                 wavelength=None,
                 **kwargs):
        super().__init__(x, data, exp_no, params, deconv,
                         tau_inf, GVD_corrected, wavelength, **kwargs)
        self.fit_type = 'Exponential'
        # self._update_progress_result(self.params)

    def pre_fit(self):
        """
        Method that optimized the pre_exponential factors trace by trace without
        optimizing the decay times (taus). It is automatically ran before a
        global fit.
        """
        fit_params = self.params.copy()
        ndata, nx = self.data.shape
        # range is descending just for no specific reason
        for iy in range(nx, 0, -1):
            # print(iy)
            single_param = lmfit.Parameters()
            single_param['y0_%i' % iy] = fit_params['y0_%i' % iy]
            single_param.add(('t0_%i' % iy), value=fit_params['t0_1'].value,
                             expr=None, vary=fit_params['t0_1'].vary)
            if self.deconv:
                single_param['fwhm_%i' % iy] = fit_params['fwhm_1']
                if self.tau_inf is not None:
                    single_param['yinf_%i' % iy] = fit_params['yinf_%i' % iy]
            for i in range(self.exp_no):
                single_param.add(('tau%i_' % (i + 1) + str(iy)),
                                 value=fit_params['tau%i_1' % (i + 1)].value,
                                 expr=None, vary=False)
                single_param.add(('pre_exp%i_' % (i + 1) + str(iy)),
                                 value=fit_params['pre_exp%i_' % (i + 1)
                                                  + str(iy)].value,
                                 vary=True)
            if self.deconv:
                result = lmfit.minimize(self._single_fit, single_param,
                                        args=(self.expNGaussDataset, iy - 1),
                                        nan_policy='propagate')
            else:
                result = lmfit.minimize(self._single_fit, single_param,
                                        args=(self.expNDataset, iy - 1),
                                        nan_policy='propagate')
            fit_params['y0_%i' % iy] = result.params['y0_%i' % iy]
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' % (i + 1) + str(iy)] = \
                    result.params['pre_exp%i_' % (i + 1) + str(iy)]
            if self.deconv:
                if self.GVD_corrected is False:
                    fit_params['t0_%i' % iy] = result.params['t0_%i' % iy]
                if self.tau_inf is not None:
                    fit_params['yinf_%i' % iy] = result.params['yinf_%i' % iy]
            self.params = fit_params
            self._prefit_done = True

    def _objective(self, params):
        """
        The optimizing function that is minimized. Is constructed to return a
        flat array  of residues, which corresponds to the data minus the
        exponential model.
        """
        if self.deconv:
            if self.GVD_corrected:
                t0 = params['t0_1'].value
                fwhm = params['fwhm_1'].value
                values = [params['tau%i_1' % (ii + 1)].value for ii in
                          range(self.exp_no)]
                if self.tau_inf is not None:
                    values.append(self.tau_inf)
                expvects = [self.expGauss(self.x - t0, tau, fwhm / 2.35482)
                            for tau in values]
                resid = self._generate_residues(self.expNGaussDatasetFast,
                                                params, expvects)
            else:
                resid = self._generate_residues(self.expNGaussDataset,
                                                params)
        else:
            t0 = params['t0_1'].value
            index = np.argmin([abs(i - t0) for i in self.x])
            values = [params['tau%i_1' % (ii + 1)].value
                      for ii in range(self.exp_no)]
            expvects = [self.exp1(self.x - t0, tau)
                        for tau in values]
            resid = self._generate_residues(self.expNDatasetFast, params,
                                            expvects)[index:, :]

        self._number_it = self._number_it + 1
        return resid.flatten()

    def global_fit(self, vary_taus=True, maxfev=None, time_constraint=False,
                   apply_weights=False, method='leastsq', **kws):
        """
        Method to fit the data to a model. Returns a modified lmfit result
        object.

        Parameters
        ----------

        vary_taus: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should
            be a list of bool equal with len equal to the number of taus.
            Each entry defines if a initial taus should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        time_constraint: bool (default False)
            If True and there are more than one tau to optimized force:
            tau2 > tau1, tau3 > tau2 and so on
            If self.deconv and True a Gaussian modified exponential model is
            applied and tau1 > fwhm.

        apply_weights: bool (default False)
            If True and weights have been defined, this will be applied in the
            fit (for defining weights) check the function define_weights.

        method: default (leastsq)
            Any valid method that can be used by lmfit minimize function

        kws:
            Any valid kwarg that can be used by lmfit minimize function
        """
        if type(vary_taus) == bool:
            vary_taus = [vary_taus for i in range(self.exp_no)]
        for i in range(self.exp_no):
            self.params['tau%i_1' % (i + 1)].vary = vary_taus[i]
        if time_constraint:
            self._apply_time_constraint()
        result = super().global_fit(maxfev=maxfev, apply_weights=apply_weights,
                                    method=method, **kws)
        if time_constraint:
            result.details['time_constraint'] = True
            self._unconstraint_times()
        return result

    def _apply_time_constraint(self):
        """
        Apply a time constraint to the "tau" values where the minimum value is
        must be higher than the previous: tau2 > tau1; tau3 > tau2.
        In case there is deconvolution tau1 > fwhm.
        """
        for i in range(self.exp_no):
            if i == 0 and self.deconv:
                self.params['tau%i_1' % (i + 1)].min = \
                    self.params['fwhm_1'].value
            elif i >= 1:
                self.params['tau%i_1' % (i + 1)].min = \
                    self.params['tau%i_1' % i].value

    def _unconstraint_times(self):
        """
        Undo the time constraint to the "tau" values
        """
        for i in range(self.exp_no):
            if i == 0 and self.deconv:
                self.params['tau%i_1' % (i + 1)].min = None
            else:
                self.params['tau%i_1' % (i + 1)].min = None


class GlobalFitTarget(GlobalFit):
    """
    Class that does a global target fit to a kinetic model. This class is uses
    a global fit evaluate the times from all the traces (kinetic constants are
    globally fitted), while the pre_exponential values (pre_exp) are estimated
    independently from each trace. The pre_exp values give later the species
    associated spectra (SAS). The Class do not generates the parameters
    automatically.
    Attributes
    ----------
    x: 1darray
       x-vector, normally time vector
    data: 2darray
       array containing the data, the number of rows should be equal to the
       len(x)
    exp_no: int
       number of exponential that will be used to fit the data.
    params: lmfit parameters object
       object containing the initial parameters values used to build an
       exponential model. These parameters are iteratively optimize to
       reduce the residual matrix formed by data-model (error matrix)
       using Levenberg-Marquardt algorithm.
    deconv: bool
       If True the fitting functions will search for the deconvolution
       parameter ("fwhm") in the params attribute, and the the model is a
       weighted sum of Gauss modified exponential functions. If False the
       the model is a weighted sum of exponential functions, and params
       should not contain the fwhm entrance.
    GVD_corrected: bool
       Defines if the chrip or group velocity dispersion (GVD) has been
       corrected. If True t0 is globally optimized (faster). If False t0 is
       separately optimized for each trace (slower). Notice in some cases
       even if the chirp or GVD has been corrected, very small variations
       of t0 that might be imperceptible and small may generate problems in
       the fit, setting this parameter to False can help to acquire
       overcome this problem although the fit will take longer.
       (only affects if deconv is True)
    weights: dictionary
       This dictionary controls if the fitting weights are applied,
       the keys are:
       "apply": Bool
       "vector": weighting vector.
       'type': type of vector defined in the ,
       'range': time range according to x-vector of the weights that are
                different than 1
       'value': val weighting value
       The dictionary keys can be passes as kwargs when instantiating the
       object
    """
    def __init__(self,
                 x,
                 data,
                 exp_no,
                 params,
                 deconv=True,
                 GVD_corrected=True,
                 wavelength=None,
                 **kwargs):
        """
        constructor function:
        Parameters
        ----------
        x: 1darray
            x-vector, normally time vector
        data: 2darray
            Array containing the data, the number of rows should be equal to
            the len(x)
        exp_no: int
            Number of exponential that will be used to fit the data
        params: lmfit parameter object
            parameters object containing the initial estimations values for all
            the parameters together with the minimum maximum and constraints.
            This object can easily be generated with GlobalTargetParams class,
            and the target Model class.
        deconv: bool (default True)
            If True the fitting functions will search for the deconvolution
            parameter ("fwhm") in the params attribute, and the the model is a
            weighted sum of Gauss modified exponential functions. If False the
            the model is a weighted sum of exponential functions, and params
            should not contain the fwhm entrance.
        GVD_corrected: bool (defautl True)
            Defines if the chrip or group velocity dispersion (GVD) has been
            corrected. If True t0 is globally optimized (faster). If False t0 is
            separately optimized for each trace (slower). Notice in some cases
            even if the chirp or GVD has been corrected, very small variations
            of t0 that might be imperceptible and small may generate problems in
            the fit, setting this parameter to False can help to overcome this
            problem although the fit will take longer.
            (only affects if deconv is True)
        kwargs:
            Related for applying weight to the fit. The dictionary obtained from
            the function define_weights can be directly pass as *+weights
        """
        super().__init__(x, data, exp_no, params, deconv,
                         None, GVD_corrected, wavelength, **kwargs)
        self.fit_type = 'Target'
        # self._update_progress_result(self.params)

    def pre_fit(self):
        """
        Method that optimized the pre_exponential factors trace by trace without
        optimizing the kinetic constants times. It is automatically ran before a
        global fit.
        """
        # initiate self.data_before_last_Fit copying from self.data which
        # will be used to fit
        # parameters have been created with lenght of self.data
        # this allow to keep after the fit a copy of the data that was fitted
        fit_params = self.params.copy()
        ndata, nx = self.data.shape
        coeffs, eigs, eigenmatrix = solve_kmatrix(self.exp_no, fit_params)
        for iy in range(nx):
            print(iy)
            single_param = lmfit.Parameters()
            for i in range(self.exp_no):
                single_param['pre_exp%i_' % (i + 1) + str(iy + 1)] = \
                    fit_params['pre_exp%i_' % (i + 1) + str(iy + 1)]
            single_param['y0_%i' % (iy + 1)] = fit_params[
                'y0_%i' % (iy + 1)]
            single_param.add(('t0_%i' % (iy + 1)),
                             value=fit_params['t0_1'].value,
                             expr=None, vary=fit_params['t0_1'].vary)
            if self.deconv:
                single_param.add(('fwhm_%i' % (iy + 1)),
                                 value=fit_params['fwhm_1'].value,
                                 expr=None,
                                 vary=fit_params['fwhm_1'].vary)

            result = lmfit.minimize(self._single_fit,
                                    single_param,
                                    args=(self.expNGaussDatasetTM, iy,
                                          [coeffs, eigs, eigenmatrix]),
                                    nan_policy='propagate')
            if not self.GVD_corrected and self.deconv:
                fit_params['t0_%i' % (iy + 1)] = result.params[
                    't0_%i' % (iy + 1)]
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' % (i + 1) + str(iy + 1)] = \
                    result.params['pre_exp%i_' % (i + 1) + str(iy + 1)]
            self.params = fit_params
        self._prefit_done = True

    def _objective(self, params, shared_t0=True):
        """
        The optimizing function that is minimized. Is constructed to return a
        flat array  of residues, which corresponds to the data minus the
        kinetic target model.
        """
        # size of the matrix = no of exponenses = no of species
        ksize = self.exp_no
        coeffs, eigs, eigenmatrix = solve_kmatrix(ksize, params)
        if self.deconv:
            if shared_t0:
                t0 = params['t0_1'].value
                fwhm = params['fwhm_1'].value / 2.35482
                expvects = [
                    coeffs[i] * self.expGauss(self.x - t0, -1 / eigs[i],
                                              fwhm)
                    for i in range(len(eigs))]
                concentrations = [sum([eigenmatrix[i, j] * expvects[j]
                                       for j in range(len(eigs))])
                                  for i in range(len(eigs))]
                resid = self._generate_residues(self.expNGaussDatasetFast,
                                                params, concentrations)
            else:
                resid = self._generate_residues(self.expNGaussDatasetTM,
                                                params,
                                                (coeffs, eigs, eigenmatrix))
        else:
            t0 = params['t0_1'].value
            index = np.argmin([abs(i - t0) for i in self.x])
            values = [params['tau%i_1' % (ii + 1)].value
                      for ii in range(self.exp_no)]
            expvects = [self.exp1(self.x - t0, tau) for tau in values]
            resid = self._generate_residues(self.expNGaussDatasetTM, params,
                                            (coeffs, eigs, eigenmatrix))[
                    index:,
                    :]
        self._number_it = self._number_it + 1
        return resid.flatten()


class GlobalFitWithIRF(GlobalFit):
    """
    Class that does a global fit using exponential models convolved with an
    instrument response function (IRF) array. A global fit evaluate the times
    from all the traces (taus are globally fitted), while the pre_exponential
    values (pre_exp) are estimated independently from each trace.
    The Class do not generates the parameters automatically.

    Attributes
    ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            array containing the data, the number of rows should be equal to the
            len(x)

        exp_no: int
            number of exponential that will be used to fit the data.

        params: lmfit parameters object
            object containing the initial parameters values used to build an
            exponential model. These parameters are iteratively optimize to
            reduce the residual matrix formed by data-model (error matrix)
            using Levenberg-Marquardt algorithm.


        """

    def __init__(self,
                 x,
                 data,
                 irf,
                 exp_no,
                 params,
                 wavelength=None):
        super().__init__(x, data, exp_no, params, True,
                         None, False, wavelength)
        self.fit_type = 'Exponential convolved'
        self.deconv = irf
        self.weights = {'apply': True, 'vector': data[:,1],
                        'range': [x[0], x[-1]],
                        'type': 'poison', 'value': None}
        self.params['t0_1'].vary = True
        self.params['t0_1'].max = 0.5
        self.params['t0_1'].min = 0
        # self._update_progress_result(self.params)

    def pre_fit(self):
        """
        Method that optimized the pre_exponential factors trace by trace without
        optimizing the decay times (taus). It is automatically ran before a
        global fit.
        """
        fit_params = self.params.copy()
        ndata, nx = self.data.shape
        # range is descending just for no specific reason
        for iy in range(nx, 0, -1):
            # print(iy)
            single_param = lmfit.Parameters()
            single_param['y0_%i' % iy] = fit_params['y0_%i' % iy]
            single_param.add(('t0_%i' % iy), value=fit_params['t0_1'].value,
                             expr=None, vary=fit_params['t0_1'].vary)

            for i in range(self.exp_no):
                single_param.add(('tau%i_' % (i + 1) + str(iy)),
                                 value=fit_params['tau%i_1' % (i + 1)].value,
                                 expr=None, vary=False)
                single_param.add(('pre_exp%i_' % (i + 1) + str(iy)),
                                 value=fit_params['pre_exp%i_' % (i + 1)
                                                  + str(iy)].value*10,
                                 vary=True)

            result = lmfit.minimize(self._single_fit, single_param,
                                    args=(self.expNDatasetIRF, iy-1, self.deconv),
                                    nan_policy='propagate')
            fit_params['y0_%i' % iy] = result.params['y0_%i' % iy]
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' % (i + 1) + str(iy)] = \
                    result.params['pre_exp%i_' % (i + 1) + str(iy)]
            self.params = fit_params
            self._prefit_done = True

    def global_fit(self, vary_taus=True, maxfev=None, time_constraint=False,
                   method='leastsq', **kws):
        """
        Method to fit the data to a model. Returns a modified lmfit result
        object.

        Parameters
        ----------

        vary_taus: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should
            be a list of bool equal with len equal to the number of taus.
            Each entry defines if a initial taus should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        time_constraint: bool (default False)
            If True and there are more than one tau to optimized force:
            tau2 > tau1, tau3 > tau2 and so on
            If self.deconv and True a Gaussian modified exponential model is
            applied and tau1 > fwhm.

        method: default (leastsq)
            Any valid method that can be used by lmfit minimize function

        kws:
            Any valid kwarg that can be used by lmfit minimize function
        """
        if type(vary_taus) == bool:
            vary_taus = [vary_taus for i in range(self.exp_no)]
        for i in range(self.exp_no):
            self.params['tau%i_1' % (i + 1)].vary = vary_taus[i]
        if time_constraint:
            self._apply_time_constraint()
        result = super().global_fit(maxfev=None, apply_weights=True,
                                    method='leastsq', **kws)
        if time_constraint:
            result.details['time_constraint'] = True
            self._unconstraint_times()
        return result

    def _generate_residues(self, function, params, extra_param):
        """
        Generate a single residue for one trace (used by global_fit)
        """
        ndata, nx = self.data.shape
        data = self.data[:]
        resid = data * 1.0
        # t0 = params['t0_1'].value
        # index =  np.argmin([abs(i - t0) for i in self.x])
        for i in range(nx):
            # resid[index:, i] = data[index:, i] - function(params, i, extra_param)
            resid[:, i] = data[:, i] - function(params, i, extra_param)
            if self.weights['apply']:
                 w = 1/np.sqrt(data[:, i])
                 w[w == np.inf] = 0
                 resid[:, i] = resid[:, i] * w
        return resid

    def _objective(self, params):
        """
        The optimizing function that is minimized. Is constructed to return a
        flat array  of residues, which corresponds to the data minus the
        exponential model.
        """
        resid = self._generate_residues(self.expNDatasetIRF,
                                        params, self.deconv)
        self._number_it = self._number_it + 1
        return resid.flatten()

    def _apply_time_constraint(self):
        """
        Apply a time constraint to the "tau" values where the minimum value is
        must be higher than the previous: tau2 > tau1; tau3 > tau2.
        In case there is deconvolution tau1 > fwhm.
        """
        for i in range(1, self.exp_no):
            self.params['tau%i_1' % (i + 1)].min = \
                self.params['tau%i_1' % i].value

    def _unconstraint_times(self):
        """
        Undo the time constraint to the "tau" values
        """
        for i in range(1, self.exp_no):
            self.params['tau%i_1' % (i + 1)].min = None

    def global_fit(self, vary_taus=True, maxfev=None, time_constraint=False,
                   method='leastsq', **kws):
        """
        Method to fit the data to a model. Returns a modified lmfit result
        object.

        Parameters
        ----------

        vary_taus: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should
            be a list of bool equal with len equal to the number of taus.
            Each entry defines if a initial taus should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        time_constraint: bool (default False)
            If True and there are more than one tau to optimized force:
            tau2 > tau1, tau3 > tau2 and so on
            If self.deconv and True a Gaussian modified exponential model is
            applied and tau1 > fwhm.

        method: default (leastsq)
            Any valid method that can be used by lmfit minimize function

        kws:
            Any valid kwarg that can be used by lmfit minimize function
        """
        if type(vary_taus) == bool:
            vary_taus = [vary_taus for i in range(self.exp_no)]
        for i in range(self.exp_no):
            self.params['tau%i_1' % (i + 1)].vary = vary_taus[i]
        if time_constraint:
            self._apply_time_constraint()
        result = super().global_fit(maxfev=None, apply_weights=True,
                                    method='leastsq', **kws)
        if time_constraint:
            result.details['time_constraint'] = True
            self._unconstraint_times()
        return result