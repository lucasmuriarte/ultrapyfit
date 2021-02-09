# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:12:36 2020

@author: 79344
"""
import numpy as np
import lmfit
from ultrafast.ModelCreatorClass import ModelCreator


class GlobalFitTargetModel(lmfit.Minimizer, ModelCreator):
    def __init__(
            self,
            x,
            data,
            exp_no,
            params,
            deconv=True,
            SVD=False,
            GVD_corrected=True,
            **kwargs):
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
        self.SVD_fit = SVD
        self.deconv = deconv
        self.exp_no = exp_no
        self.GVD_corrected = GVD_corrected
        self._number_it = 0
        ModelCreator.__init__(self, self.exp_no, self.x, None)
        lmfit.Minimizer.__init__(
            self,
            self.objectiveTagetFit,
            params,
            nan_policy='propagate')

    def singleFit(self, params, function, i, extra_params):
        """does a fit of a single trace"""
        if self.deconv:
            return self.data[:, i] - function(params, i, extra_params)
        else:
            t0 = params['t0_%i' % (i + 1)].value
            index = np.argmin([abs(i - t0) for i in self.x])
            return self.data[index:, i] - function(params, i, extra_params)

    def objectiveTagetFit(params, shared_t0=True):
        ksize = self.exp_no  # size of the matrix = no of exponenses = no of species
        kmatrix = np.array(
            [[params['k_%i%i' % (i + 1, j + 1)].value for j in range(ksize)] for i in range(ksize)])
        cinitials = [params['c_%i' % (i + 1)].value for i in range(ksize)]
        eigs, vects = np.linalg.eig(kmatrix)  # do the eigenshit
        eigenmatrix = np.array(vects)

        coeffs = np.linalg.solve(eigenmatrix, cinitials)
        if self.deconv:
            if shared_t0:
                t0 = params['t0_1'].value
                fwhm = params['fwhm_1'].value
                expvects = [coeffs[i] *
                            self.expGauss(self.x -
                                          t0, -
                                          1 /
                                          eigs[i], fwhm /
                                          2.35482) for i in range(len(eigs))]
                concentrations = [sum([eigenmatrix[i, j] * expvects[j]
                                       for j in range(len(eigs))]) for i in range(len(eigs))]
                resid = self.generateResidues(
                    self.expNGaussDatasetFast, params, concentrations)
            else:
                resid = self.generateResidues(
                    expNGaussDatasetTM, params, (coeffs, eigs, eigenmatrix))
        else:
            t0 = params['t0_1'].value
            index = np.argmin([abs(i - t0) for i in self.x])
            values = [params['tau%i_1' %
                             (ii + 1)].value for ii in range(self.exp_no)]
            expvects = [self.exp1(self.x - t0, tau) for tau in values]
            resid = self.generateResidues(
                self.expNGaussDatasetTM, params, (coeffs, eigs, eigenmatrix))[
                index:, :]
        self._number_it = self._number_it + 1
        if(self.number_it % 100 == 0):
            print(self.number_it)
            print(sum(np.abs(resid.flatten())))
        return resid.flatten()

    def generateResidues(self, funtion, params, extra_para):
        ndata, nx = self.data.shape
        data = self.data[:]
        resid = 0.0 * data[:]
        if extra_param is not None:
            for i in range(nx):
                resid[:, i] = data[:, i] - funct(params, i, extra_param)
        if self.weights['apply']:
            resid[:, i] = resid[:, i] * self.weights['vector']
        return resid

    def preFit(self):
        # initiate self.data_before_last_Fit copying from self.data which will be used to fit
        # parameters have been created with lenght of self.data
        # this allow to keep after the fit a copy of the data that was fitted
        fit_params = self.initial_params.copy()
        ndata, nx = self.data_before_last_Fit.shape
        ksize = self.exp_no  # size of the matrix = no of exponenses = no of species
        kmatrix = np.array([[fit_params['k_%i%i' % (
            i + 1, j + 1)].value for j in range(ksize)] for i in range(ksize)])
        cinitials = [fit_params['c_%i' % (i + 1)].value for i in range(ksize)]
        eigs, vects = np.linalg.eig(kmatrix)  # do the eigenshit
        # eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))])
        eigenmatrix = np.array(vects)
        # solve the initial conditions sheet
        coeffs = np.linalg.solve(eigenmatrix, cinitials)
        # didnt tested but should work, if no then probably minor correction is
        # needed.
        for iy in range(nx):
            print(iy)
            single_param = lmfit.Parameters()
            for i in range(self.exp_no):
                single_param['pre_exp%i_' %
                             (i + 1) + str(iy + 1)] = fit_params['pre_exp%i_' %
                                                                 (i + 1) + str(iy + 1)]
            single_param['y0_%i' % (iy + 1)] = fit_params['y0_%i' % (iy + 1)]
            single_param.add(
                ('t0_%i' % (iy + 1)),
                value=fit_params['t0_1'].value,
                expr=None,
                vary=self.t0_vary)
            if self.deconv:
                single_param.add(
                    ('fwhm_%i' % (iy + 1)),
                    value=fit_params['fwhm_1'].value,
                    expr=None,
                    vary=fit_params['fwhm_1'].vary)
            result = lmfit.minimize(
                self.single_fit, single_param, args=(
                    self.expNGaussDatasetTM, iy, [
                        coeffs, eigs, eigenmatrix]), nan_policy='propagate')
            if self.GVD_correction == False and self.deconv:
                fit_params['t0_%i' %
                           (iy + 1)] = result.params['t0_%i' %
                                                     (iy + 1)]
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' %
                           (i + 1) + str(iy + 1)] = result.params['pre_exp%i_' %
                                                                  (i + 1) + str(iy + 1)]
            self.params = fit_params
        self.prefit_done = True

    def finalFit(self, vary_taus=True, maxfev=None, apply_weights=False):
        if isinstance(vary_taus, bool):
            vary_taus = [vary_taus for i in range(self.exp_no)]
        self.Fit_completed = False
        if not self.prefit_done:
            self.preFit()
        # self.type_fit is important to know if we are doing an expoential or
        # taget fit
        fit_condition = [maxfev, 'No constrain', self.type_fit]
        fit_params = self.params
        if apply_weights and len(self.weights['vector']) == len(self.x):
            self.weights['apply'] = True
            fit_condition.append(self.weights)
        else:
            fit_condition.append('no weights')
        if maxfev is not None:
            self.resultados = self.minimize(params=fit_params, maxfev=maxfev)
        else:
            self.resultados = self.minimize(params=fit_params)
        self.number_it = 0
        self.Fit_completed = True
        if isinstance(fit_condition[3], dict):
            self.weights['apply'] = False
