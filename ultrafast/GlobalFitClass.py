# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:00:23 2020

@author: lucas
"""
import numpy as np
import lmfit
from ultrafast.ModelCreatorClass import ModelCreator
from ultrafast.GlobExpParams import GlobExpParameters


def globalfit_exponential(
        x,
        data,
        *taus,
        vary=True,
        t0=0,
        maxfev=5000,
        **kwargs):
    """
    Function that does a global fit of a weighted sum of exponential equal to the number
    of time constants (taus) pass. It gives a result object which is a modified object
    with added properties. The result can be evaluated with the explore result class.

    A global fit evaluate the times from all the traces (taus are globally fitted), while
    the pre_exponential values (pre_exp) are estimated independently from each trace. The
    pre_exp values give later the decay associated spectra (DAS). The function generates the
    parameters automatically.

    Parameters
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2d array
        Array containing the data, the number of rows should be equal to the len(x)

    taus: float or int
        Initial estimates for the exponential decay times. There is no limit of tau numbers.
        each tau correspond to an exponential function. Is recommended to use the minimum
        number of exponential that correctly describe the data set.

    vary: bool or list of bool
        If True or False all taus are optimized or fixed. If a list, should be a list of bool
        equal with len equal to the number of taus. Each entry defines if a initial taus
        should be optimized or not.

    t0: int
        Initial time value where to start evaluating the function. Only x vector values
        higher than t0 will be take into consideration.

    maxfev: int (default 5000)
        Maximum number of iterations of the fit.

    kwargs:
        Related for applying weight to the fit. The dictionary obtained from the function
        define_weights can be directly pass as *+weights

    e.g.1: globalFitSumExponential(x, data, 8,30, 200, vary=True, t0=3)
        A weighted sum of three exponential function will be fitted to the data set
        the initial estimates are 8, 30 and 200, where all will be optimized. The
        function is only evaluated from x values higher than 3

    e.g.2: globalFitSumExponential(x, data, 8,30, 200, vary=[True, False, True], t0=0)
        A weighted sum of three exponential function will be fitted to the data set
        the initial estimates are 8, 30 and 200 where 8 and 200 are optimized and 30 is fixed.
        function is only evaluated from x values higher than 0
    """
    taus = list(taus)
    if isinstance(vary, bool):
        vary = [vary for i in taus]
    exp_no = len(taus)
    _, n_traces = data.shape
    params = GlobExpParameters(n_traces, taus)
    params.adjustParams(t0, False, None)
    fit = GlobalFitExponential(
        x,
        data,
        exp_no,
        params.params,
        deconv=False,
        **kwargs)
    results = fit.finalFit(vary_taus=vary, maxfev=maxfev)
    return results


def globalfit_gauss_exponential(
        x,
        data,
        *taus,
        vary=True,
        fwhm=0.12,
        tau_inf=1E12,
        t0=0,
        vary_t0=True,
        vary_fwhm=False,
        maxfev=5000,
        GVD_corrected=True,
        **kwargs):
    """
    Function that does a global fit of a weighted sum of gaussian modified exponential
    equal to the number of time constants (taus) pass. It gives a result object which is a
    modified object with added properties. The result can be evaluated with the explore
    result class.

    A global fit evaluate the times from all the traces (taus are globally fitted), while
    the pre_exponential values (pre_exp) are estimated independently from each trace. The
    pre_exp values give later the decay associated spectra (DAS). The function generates the
    parameters automatically.

    Parameters
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2d array
        Array containing the data, the number of rows should be equal to the len(x)

    taus: float or int
        Initial estimates for the exponential decay times. There is no limit of tau numbers.
        each tau correspond to an exponential function. Is recommended to use the minimum
        number of exponential that correctly describe the data set.

    vary: bool or list of bool
        If True or False all taus are optimized or fixed. If a list, should be a list of bool
        equal with len equal to the number of taus. Each entry defines if a initial taus
        should be optimized or not.

    fwhm: int or float (default 0.12)
        Full width half maximum of the laser pulse length use in the experiment. This value
        is used to to fit a modified gaussian exponential function, and can be determined
        with an external measurement or for example in case of TRUV-vis data fitting the stimulated
        raman signal of the solvent to a gaussian function.

    tau_inf: float or int or None (default 1E12)
        An extra decay time use to evaluate possible photoproduct. This should be used if the signal
        at long delay times is not completely recovered. If the signal at long delay times is zero
        tau_inf should be set to None.

    t0: int (default 0)
        t0 in a modified gaussian exponential fit defines the point where the signal start raising.
        This should be the 0, and is the time defining the time where the pump and probe overlaps
        (notice is different from the t0 in a simple exponential fit)

    vary_t0: bool (default True)
        Defines if the t0 is optimized. We recommend to always set this value to true

    vary_fwhm: bool (default False)
        Defines if the fwhm is optimized. We recommend to always set this value to False, and determine this
        value from an external measurement

    GVD_corrected: bool (defautl True)
        Defines if the chrip or group velocity dispersion (GVD) has been corrected. If True t0 is globally
        optimized (faster). If False t0 is separately optimized for each trace (slower). Notice in some cases
        even if the chirp or GVD has been corrected, very small variations of t0 that might be imperceptible
        and small may generate problems in the fit, setting this parameter to False can help to acquire overcome
        this problem although the fit will take longer.

    maxfev: int (default 5000)
        Maximum number of iterations of the fit.

    kwargs:
        Related for applying weight to the fit. The dictionary obtained from the function
        define_weights can be directly pass as *+weights

    e.g.1: globalFitSumExponential(x, data, 8,30, 200, vary=True, fwhm=0.16)
        A weighted sum of three exponential function will be fitted to the data set
        the initial estimates are 8, 30 and 200, where all will be optimized.he fwhm
        of the Gauss function is 0.16.

    e.g.2: globalFitSumExponential(x, data, 8,30, 200, vary=[True, False, True], fwhm=0.12)
        A weighted sum of three Gauss modified exponential function will be fitted to the data set
        the initial estimates are 8, 30 and 200 where 8 and 200 are optimized and 30 is fixed. The fwhm
        of the Gauss function is 0.12.
    """
    taus = list(taus)
    if isinstance(vary, bool):
        vary = [vary for i in taus]
    exp_no = len(taus)
    _, n_traces = data.shape
    params = GlobExpParameters(n_traces, taus)
    params.adjustParams(t0, vary_t0, fwhm, vary_fwhm, GVD_corrected, tau_inf)
    fit = GlobalFitExponential(
        x,
        data,
        exp_no,
        params.params,
        vary=True,
        deconv=True,
        tau_inf=tau_inf,
        GVD_corrected=GVD_corrected,
        **kwargs)
    results = fit.finalFit(vary_taus=vary, maxfev=maxfev)
    return results


class GlobalFitExponential(lmfit.Minimizer, ModelCreator):
    """
    Class that does a global fit using exponential models. This class is uses by the function,
    globalfit_gauss_exponential and globalfit_exponential functions. A global fit evaluate the
    times from all the traces (taus are globally fitted), while the pre_exponential values
    (pre_exp) are estimated independently from each trace. The pre_exp values give later the
    decay associated spectra (DAS). The Class do not generates the parameters automatically.

    Attributes
    ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            array containing the data, the number of rows should be equal to the len(x)

        exp_no: int
            number of exponential that will be used to fit the data.

        params: lmfit parameters object
            object containing the initial parameters values used to build an exponential model.
            These parameters are iteratively optimize to reduce the residual matrix formed by
            data-model (error matrix) using Levenberg-Marquardt algorithm.

        deconv: bool
            If True the fitting functions will search for the deconvolution parameter ("fwhm") in the
            params attribute, and the the model is a weighted sum of Gauss modified exponential functions.
            If False the the model is a weighted sum of exponential functions, and params should not contain
            the fwhm entrance.

        tau_inf: float or int or None
            An extra decay time use to evaluate possible photoproduct. This should be used if the signal
            at long delay times is not completely recovered and if deconv is set to True. If the signal
            at long delay times is zero tau_inf should be set to None. (only affects if deconv is True)

        GVD_corrected: bool
            Defines if the chrip or group velocity dispersion (GVD) has been corrected. If True t0 is globally
            optimized (faster). If False t0 is separately optimized for each trace (slower). Notice in some cases
            even if the chirp or GVD has been corrected, very small variations of t0 that might be imperceptible
            and small may generate problems in the fit, setting this parameter to False can help to acquire overcome
            this problem although the fit will take longer. (only affects if deconv is True)

        weights: dictionary
            This dictionary controls if the fitting weights are applied, the keys are:
            "apply": Bool
            "vector": weighting vector.
            'type': type of vector defined in the ,
            'range': time range according to x-vector of the weights that are different than 1
            'value': val weighting value
            The dictionary keys can be passes as kwargs when instantiating the object
    """

    def __init__(
            self,
            x,
            data,
            exp_no,
            params,
            deconv=True,
            tau_inf=1E+12,
            GVD_corrected=True,
            **kwargs):
        """
        constructor function:

        Parameters
        ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            Array containing the data, the number of rows should be equal to the len(x)

        exp_no: int
            Number of exponential that will be used to fit the data

        deconv: bool (default True)
            If True the fitting functions will search for the deconvolution parameter ("fwhm") in the
            params attribute, and the the model is a weighted sum of Gauss modified exponential functions.
            If False the the model is a weighted sum of exponential functions, and params should not contain
            the fwhm entrance.

        tau_inf: float or int or None (default 1E12)
            An extra decay time use to evaluate possible photoproduct. This should be used if the signal
            at long delay times is not completely recovered and if deconv is set to True. If the signal
            at long delay times is zero tau_inf should be set to None. (only affects if deconv is True)

        GVD_corrected: bool (defautl True)
            Defines if the chrip or group velocity dispersion (GVD) has been corrected. If True t0 is globally
            optimized (faster). If False t0 is separately optimized for each trace (slower). Notice in some cases
            even if the chirp or GVD has been corrected, very small variations of t0 that might be imperceptible
            and small may generate problems in the fit, setting this parameter to False can help to acquire overcome
            this problem although the fit will take longer. (only affects if deconv is True)

        kwargs:
            Related for applying weight to the fit. The dictionary obtained from the function
            define_weights can be directly pass as *+weights
        """
        # default kwargs
        weights = dict({'apply': False,
                        'vector': None,
                        'range': [],
                        'type': 'constant',
                        'value': 2},
                       **kwargs)
        self.weights = weights
        self.x = x
        self.data = data
        self.params = params
        self.deconv = deconv
        self.tau_inf = tau_inf
        self.exp_no = exp_no
        self.GVD_corrected = GVD_corrected
        self._number_it = 0
        self._prefit_done = False
        self.fit_completed = False
        ModelCreator.__init__(self, self.exp_no, self.x, self.tau_inf)
        lmfit.Minimizer.__init__(
            self,
            self._objectiveExponential,
            params,
            nan_policy='propagate')

    def preFit(self):
        """
        Method that optimized the pre_exponential factors trace by trace without optimizing
        the decay times (taus). It is automatically ran before the final fit.
        """
        fit_params = self.params.copy()
        ndata, nx = self.data.shape
        # range is descending just for no specific reason
        for iy in range(nx, 0, -1):
            print(iy)
            single_param = lmfit.Parameters()
            single_param['y0_%i' % iy] = fit_params['y0_%i' % iy]
            single_param.add(
                ('t0_%i' % iy),
                value=fit_params['t0_1'].value,
                expr=None,
                vary=fit_params['t0_1'].vary)
            if self.deconv:
                single_param['fwhm_%i' % iy] = fit_params['fwhm_1']
                if self.tau_inf is not None:
                    single_param['yinf_%i' % iy] = fit_params['yinf_%i' % iy]
            for i in range(self.exp_no):
                single_param.add(('tau%i_' %
                                  (i + 1) + str(iy)), value=fit_params['tau%i_1' %
                                                                       (i + 1)].value, expr=None, vary=False)
                single_param.add(('pre_exp%i_' %
                                  (i + 1) + str(iy)), value=fit_params['pre_exp%i_' %
                                                                       (i + 1) + str(iy)].value, vary=True)
            if self.deconv:
                result = lmfit.minimize(self._singleFit, single_param, args=(
                    self.expNGaussDataset, iy - 1), nan_policy='propagate')
            else:
                result = lmfit.minimize(
                    self._singleFit, single_param, args=(
                        self.expNDataset, iy - 1), nan_policy='propagate')
            fit_params['y0_%i' % iy] = result.params['y0_%i' % iy]
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' %
                           (i + 1) + str(iy)] = result.params['pre_exp%i_' %
                                                              (i + 1) + str(iy)]
            if self.deconv:
                if self.GVD_corrected is False:
                    fit_params['t0_%i' % iy] = result.params['t0_%i' % iy]
                if self.tau_inf is not None:
                    fit_params['yinf_%i' % iy] = result.params['yinf_%i' % iy]
            self.params = fit_params
            self._prefit_done = True

    def finalFit(
            self,
            vary_taus=True,
            maxfev=None,
            time_constraint=False,
            apply_weights=False):
        """
        Method to fit the data to a model. Returns a modified lmfit result object.

        Parameters
        ----------

        vary_taus: bool or list of bool
            If True or False all taus are optimized or fixed. If a list, should be a list of bool
            equal with len equal to the number of taus. Each entry defines if a initial taus
            should be optimized or not.

        maxfev: int (default 5000)
            maximum number of iterations of the fit.

        time_constraint: bool (default False)
            If True and there are more than one tau to optimized force tau2 > tau1, tau3 > tau2 and so on
            If self.deconv is True a Gaussian modified exponential model is applied and tau1 > fwhm.

        apply_weights: bool (default False)
            If True and weights have been defined, this will be applied in the fit (for defining weights) check
            the function define_weights.
        """
        if isinstance(vary_taus, bool):
            vary_taus = [vary_taus for i in range(self.exp_no)]
        self.fit_completed = False
        if self._prefit_done is False:
            self.preFit()
        # self.type_fit is important to know if we are doing an exponential or target fit
        # this is used later for exploring the results
        fit_condition = [maxfev, time_constraint, 'Exponential']
        fit_params = self.params
        for i in range(self.exp_no):
            fit_params['tau%i_1' % (i + 1)].vary = vary_taus[i]
        if time_constraint:
            for i in range(self.exp_no):
                if i == 0:
                    fit_params['tau%i_1' %
                               (i + 1)].min = fit_params['fwhm_1'].value
                else:
                    fit_params['tau%i_1' %
                               (i + 1)].min = fit_params['tau%i_1' %
                                                         i].value
        if apply_weights and len(self.weights['vector']) == len(self.x):
            self.weights['apply'] = True
            fit_condition.append(self.weights)
        else:
            self.weights['apply'] = False
            fit_condition.append('no weights')
        if maxfev is not None:
            resultados = self.minimize(params=fit_params, max_nfev=maxfev)
        else:
            resultados = self.minimize(params=fit_params)
        resultados = self._addToResultados(resultados, fit_condition)
        self._number_it = 0
        self.fit_completed = True
        if isinstance(fit_condition[3], dict):
            self.weights['apply'] = False
        return resultados

    def _addToResultados(self, resultados, fit_condition):
        """
        Add as attributes to the lmfit results object: these are the data, the time the wavelength.
        Also add fit details such as the number of exponential if convolve with a gaussian or not,
        tau_inf, maxfev and other properties that are later use by UltrafastExperiments class and
        other classes as ExploreResults.
        """
        resultados.x = self.x
        resultados.data = self.data
        resultados.wavelength = np.array(
            [i for i in range(1, self.data.shape[1] + 1)])
        tau_inf = self.tau_inf if self.deconv else None
        resultados.details = {
            'exp_no': self.exp_no,
            'deconv': self.deconv,
            'type': 'Exponential',
            'tau_inf': tau_inf,
            'maxfev': fit_condition[0],
            'time_constraint': fit_condition[1],
            'svd_fit': False,
            'derivate': False,
            'avg_traces': 'unknown'}
        resultados.weights = False if not self.weights['apply'] else self.weights
        return resultados

    def _singleFit(self, params, function, i):
        """
        Generate a single residue for one trace (used by prefit)
        """
        if self.deconv:
            return self.data[:, i] - function(params, i)
        else:
            t0 = params['t0_%i' % (i + 1)].value
            index = np.argmin([abs(i - t0) for i in self.x])
            return self.data[index:, i] - function(params, i)

    def _objectiveExponential(self, params):
        """
        The optimizing function that is minimized. Is constructed to return a flat array
        of residues, which corresponds to the data minus the exponential model.
        """
        if self.deconv:
            if self.GVD_corrected:
                t0 = params['t0_1'].value
                fwhm = params['fwhm_1'].value
                values = [params['tau%i_1' %
                                 (ii + 1)].value for ii in range(self.exp_no)]
                if self.tau_inf is not None:
                    values.append(self.tau_inf)
                expvects = [
                    self.expGauss(
                        self.x -
                        t0,
                        tau,
                        fwhm /
                        2.35482) for tau in values]
                resid = self._generateResidues(
                    self.expNGaussDatasetFast, params, expvects)
            else:
                resid = self._generateResidues(self.expNGaussDataset, params)
        else:
            t0 = params['t0_1'].value
            index = np.argmin([abs(i - t0) for i in self.x])
            values = [params['tau%i_1' %
                             (ii + 1)].value for ii in range(self.exp_no)]
            expvects = [self.exp1(self.x - t0, tau) for tau in values]
            resid = self._generateResidues(
                self.expNDatasetFast, params, expvects)[index:, :]

        self._number_it = self._number_it + 1
        if self._number_it % 100 == 0:
            print(self._number_it)
            print(sum(np.abs(resid.flatten())))
        return resid.flatten()

    def _generateResidues(self, function, params, extra_param=None):
        """
        Generate a single residue for one trace (used by finalfit)
        """
        ndata, nx = self.data.shape
        data = self.data[:]
        resid = data * 1.0
        if extra_param is not None:
            for i in range(nx):
                resid[:, i] = data[:, i] - function(params, i, extra_param)
        else:
            for i in range(nx):
                resid[:, i] = data[:, i] - function(params, i)
        if self.weights['apply']:
            resid[:, i] = resid[:, i] * self.weights['vector']
        return resid
