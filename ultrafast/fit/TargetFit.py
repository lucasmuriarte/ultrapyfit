# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:12:36 2020
@author: 79344
"""
import numpy as np
import lmfit
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.utils.divers import solve_kmatrix


class GlobalFitTargetModel(lmfit.Minimizer, ModelCreator):
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
        weights = dict({'apply': False, 'vector': None, 'range': [],
                        'type': 'constant', 'value': 2}, **kwargs)
        self.weights = weights
        self.x = x
        self.data = data
        self.params = params
        self.SVD_fit = False
        self.deconv = deconv
        self.exp_no = exp_no
        self.GVD_corrected = GVD_corrected
        self.fit_completed = False
        self._number_it = 0
        self._prefit_done = False
        ModelCreator.__init__(self, self.exp_no, self.x, None)
        lmfit.Minimizer.__init__(self, self._objectiveTarget,
                                 params, nan_policy='propagate')

    def preFit(self):
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
            single_param['y0_%i' % (iy + 1)] = fit_params['y0_%i' % (iy + 1)]
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

    def global_fit(self, maxfev=None, apply_weights=False):
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
        """
        self.fit_completed = False
        if not self._prefit_done:
            self.preFit()
        fit_condition = [maxfev, False]
        fit_params = self.params
        if apply_weights and len(self.weights['vector']) == len(self.x):
            self.weights['apply'] = True
            fit_condition.append(self.weights)
        else:
            fit_condition.append('no weights')
        if maxfev is not None:
            resultados = self.minimize(params=fit_params, maxfev=maxfev)
        else:
            resultados = self.minimize(params=fit_params)
        resultados = self._addToResultados(resultados, fit_condition)
        self._number_it = 0
        self.fit_completed = True
        if self.weights['apply']:
            self.weights['apply'] = False
        return resultados

    def _addToResultados(self, resultados, fit_condition):
        """
        Add as attributes to the lmfit results object: these are the data, the
        time the wavelength. Also add fit details such as the number of
        exponential if convolve with a gaussian or not, tau_inf, maxfev and
        other properties that are later use by UltrafastExperiments class and
        other classes as ExploreResults.
        """
        resultados.time = self.x
        resultados.data = self.data
        resultados.wavelength = np.array([i for i in
                                          range(1, self.data.shape[1] + 1)])
        resultados.details = {'exp_no': self.exp_no,
                              'deconv': self.deconv,
                              'type': 'Target',
                              'tau_inf': None,
                              'maxfev': fit_condition[0],
                              'time_constraint': fit_condition[1],
                              'svd_fit': self.SVD_fit,
                              'derivate': False,
                              'avg_traces': 'unknown'}
        if not self.weights['apply']:
            resultados.weights = False
        else:
            resultados.weights = self.weights
        return resultados

    def _single_fit(self, params, function, i, extra_params):
        """
        does a fit of a single trace use by preFit method
        """
        if self.deconv:
            return self.data[:, i] - function(params, i, extra_params)
        else:
            t0 = params['t0_%i' % (i + 1)].value
            index = np.argmin([abs(i - t0) for i in self.x])
            return self.data[index:, i] - function(params, i, extra_params)

    def _objectiveTarget(self, params, shared_t0=True):
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
                    coeffs[i] * self.expGauss(self.x - t0, -1 / eigs[i], fwhm)
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
                                            (coeffs, eigs, eigenmatrix))[index:,
                    :]
        self._number_it = self._number_it + 1
        if self._number_it % 100 == 0:
            print(self._number_it)
            print(sum(np.abs(resid.flatten())))
        return resid.flatten()

    def _generate_residues(self, funtion, params, extra_param):
        """
        Generate a single residue for one trace (used by global_fit)
        """
        ndata, nx = self.data.shape
        data = self.data[:]
        resid = 0.0 * data[:]
        for i in range(nx):
            resid[:, i] = data[:, i] - funtion(params, i, extra_param)
            if self.weights['apply']:
                resid[:, i] = resid[:, i] * self.weights['vector']
        return resid

