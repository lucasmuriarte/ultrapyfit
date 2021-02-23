# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:12:36 2020

@author: 79344
"""
import numpy as np
import lmfit
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.utils.divers import solve_kmatrix

class GlobalFitTargetModel(lmfit.Minimizer,ModelCreator):
    def __init__(self,
                 x,
                 data,
                 exp_no,
                 params,
                 deconv=True,
                 SVD=False,
                 GVD_corrected=True,
                 **kwargs):
        weights = dict({'apply': False, 'vector': None, 'range': [],
                        'type': 'constant', 'value': 2}, **kwargs)
        self.weights = weights
        self.x = x
        self.data = data
        self.params = params
        self.SVD_fit = SVD
        self.deconv = deconv
        self.exp_no = exp_no
        self.GVD_corrected = GVD_corrected
        self.fit_completed = False
        self._number_it = 0
        self._prefit_done = False
        ModelCreator.__init__(self, self.exp_no, self.x, None)
        lmfit.Minimizer.__init__(self, self._objectiveTarget,
                                 params, nan_policy='propagate')
    
    def _single_fit(self, params, function, i, extra_params):
        """
        does a fit of a single trace
        """
        if self.deconv:
            return self.data[:, i] - function(params, i, extra_params)
        else:
            t0 = params['t0_%i' % (i+1)].value
            index = np.argmin([abs(i-t0) for i in self.x])
            return self.data[index:, i] - function(params, i, extra_params)
    
    def _objectiveTarget(self, params, shared_t0=True):
        # size of the matrix = no of exponenses = no of species
        ksize = self.exp_no
        coeffs, eigs, eigenmatrix = solve_kmatrix(ksize, params)
        if self.deconv:
            if shared_t0:
                t0 = params['t0_1'].value
                fwhm = params['fwhm_1'].value/2.35482
                expvects = [coeffs[i]*self.expGauss(self.x-t0, -1/eigs[i], fwhm)
                            for i in range(len(eigs))]
                concentrations = [sum([eigenmatrix[i, j]*expvects[j]
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
            index = np.argmin([abs(i-t0) for i in self.x])
            values = [params['tau%i_1' % (ii+1)].value
                      for ii in range(self.exp_no)]
            expvects = [self.exp1(self.x-t0, tau) for tau in values]
            resid = self._generate_residues(self.expNGaussDatasetTM, params,
                                            (coeffs, eigs, eigenmatrix))[index:, :]
        self._number_it = self._number_it+1
        if self._number_it % 100 == 0:
            print(self._number_it)
            print(sum(np.abs(resid.flatten())))
        return resid.flatten()
    
    def _generate_residues(self, funtion, params, extra_param):
        ndata, nx = self.data.shape
        data = self.data[:]
        resid = 0.0*data[:]
        for i in range(nx):
            resid[:, i] = data[:, i] - funtion(params, i, extra_param)
            if self.weights['apply']:
                resid[:, i] = resid[:, i]*self.weights['vector']
        return resid
    
    def preFit(self):
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
                single_param['pre_exp%i_' % (i+1) + str(iy+1)] =\
                    fit_params['pre_exp%i_' % (i+1) + str(iy+1)]
            single_param['y0_%i' % (iy+1)] = fit_params['y0_%i' % (iy+1)]
            single_param.add(('t0_%i' % (iy+1)), value=fit_params['t0_1'].value,
                             expr=None, vary=fit_params['t0_1'].vary)
            if self.deconv:
                single_param.add(('fwhm_%i' % (iy+1)),
                                 value=fit_params['fwhm_1'].value,
                                 expr=None,
                                 vary=fit_params['fwhm_1'].vary)

            result = lmfit.minimize(self._single_fit,
                                    single_param,
                                    args=(self.expNGaussDatasetTM, iy,
                                          [coeffs, eigs, eigenmatrix]),
                                    nan_policy='propagate')
            if not self.GVD_corrected and self.deconv:
                fit_params['t0_%i' % (iy+1)] = result.params['t0_%i' % (iy+1)]
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' % (i+1) + str(iy+1)] = \
                    result.params['pre_exp%i_' % (i+1) + str(iy+1)]
            self.params = fit_params
        self._prefit_done = True

    def finalFit(self, maxfev=None, apply_weights=False):
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
        resultados.x = self.x
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
