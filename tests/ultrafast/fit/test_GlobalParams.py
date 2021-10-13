# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:10:17 2020

@author: 79344
"""

import unittest
from ultrafast.fit.GlobalParams import GlobExpParameters, GlobalTargetParameters
from parameterized import parameterized
from ultrafast.graphics.targetmodel import Model

taus = [8, 30, 200]
n_traces = 5
model = Model.load("tests/ultrafast/fit/testmodel2.model")
params_model = model.genParameters()
exp_no = params_model['exp_no']

class TestGlobExpParameters(unittest.TestCase):
    """
    test for GlobExpParameters Class
    """
    
    @parameterized.expand([[0, False], [5, True]])
    def test__generateParams(self, t0, vary):
        params = GlobExpParameters(n_traces, taus)
        params._generateParams(t0, vary)
        # two parameter per tau including t0 thus: len(taus)+1
        number = (len(taus)+1) * 2 * n_traces
        self.assertEqual(len(params.params), number)
        self.assertEqual(params.params['tau1_1'].value, taus[0])
        self.assertEqual(params.params['tau2_1'].value, taus[1])
        self.assertEqual(params.params['tau3_1'].value, taus[2])
        self.assertEqual(params.params['t0_1'].value, t0)
        self.assertEqual(params.params['t0_1'].vary, vary)

    @parameterized.expand([[0.12, False, 1E12],
                           [0.18, True, None],
                           [0.15, False, 1E8]])
    def test__addDeconvolution(self, fwhm, opt_fwhm, tau_inf):
        params = GlobExpParameters(n_traces, taus)
        params._generateParams(0, False)
        params._add_deconvolution(fwhm, opt_fwhm, tau_inf)
        # notice tau_inf is not a parameter and thus is not
        # added only its preexponential function plus fwhm
        number = (len(taus)+2) * 2 * n_traces
        if tau_inf is None:
            number -= n_traces
        self.assertEqual(len(params.params), number)
        self.assertEqual(params.params['fwhm_1'].value, fwhm)
        self.assertEqual(params.params['fwhm_1'].vary, opt_fwhm)
        self.assertEqual(params.params['fwhm_2'].expr, 'fwhm_1')

    @parameterized.expand([[0, True, 0.12, False, True, 1E12],
                           [0, False, None, True, True, None],
                           [0, True, None, True, False, None],
                           [0.5, True, 0.18, False, False, 1E12]])
    def test_adjustParams(self, t0, vary_t0, fwhm, opt_fwhm,
                          GVD_corrected, tau_inf):
        params = GlobExpParameters(n_traces, taus)
        params.adjustParams(t0, vary_t0, fwhm, opt_fwhm, 
                            GVD_corrected, tau_inf)
        self.assertEqual(params.params['tau1_1'].value, taus[0])
        self.assertEqual(params.params['tau2_1'].value, taus[1])
        self.assertEqual(params.params['tau3_1'].value, taus[2])
        self.assertEqual(params.params['tau1_2'].expr, 'tau1_1')
        self.assertEqual(params.params['tau2_2'].expr, 'tau2_1')
        self.assertEqual(params.params['tau3_2'].expr, 'tau3_1')
        self.assertEqual(params.params['t0_1'].value, t0)
        number = (len(taus)+1) * 2 * n_traces
        if fwhm is not None:
            self.assertEqual(params.params['t0_1'].vary, vary_t0)
            if tau_inf is not None:
                number = number + n_traces * 2
            else:
                number + n_traces
            self.assertEqual(params.params['fwhm_1'].vary, opt_fwhm)
            self.assertEqual(params.params['fwhm_1'].value, fwhm)
            self.assertEqual(params.params['fwhm_1'].value, fwhm)
            if GVD_corrected:
                self.assertEqual(params.params['t0_2'].expr, 't0_1')
            else:
                self.assertEqual(params.params['t0_2'].expr, None)
        else:
            # verify if there is no deconvolution t0 is fixed
            self.assertFalse(params.params['t0_1'].vary)
        self.assertEqual(len(params.params), number)

class TestGlobalTargetParams(unittest.TestCase):
    def test___init__(self):
        params = GlobalTargetParameters(n_traces, model)
        self.assertTrue(params.params == params_model)

    @parameterized.expand([[0, False, False],
                           [5, True, True],
                           [5, True, False],
                           [5, False, True]])
    def test__add_preexp_t0_y0(self, t0, vary, gvd_corrected):
        params = GlobalTargetParameters(n_traces, model)
        params._add_preexp_t0_y0(t0, vary, gvd_corrected)
        # two parameter per tau including t0 thus: len(taus)+1
        number = len(params_model) + 2 * n_traces + exp_no * (n_traces + 1)
        self.assertEqual(len(params.params), number)
        self.assertEqual(params.params['t0_1'].value, t0)
        self.assertEqual(params.params['t0_1'].vary, vary)
        if gvd_corrected:
            self.assertEqual(params.params['t0_2'].expr, 't0_1')
        else:
            self.assertTrue(params.params['t0_2'].expr is None)
    
    @parameterized.expand([[0.12, False],
                           [0.18, True],
                           [0.15, False]])
    def test__addDeconvolution(self, fwhm, opt_fwhm):
        params = GlobalTargetParameters(n_traces, model)
        params._add_deconvolution(fwhm, opt_fwhm)
        # notice tau_inf is not a parameter and thus is not
        # added only its preexponential function plus fwhm
        number = len(params_model) + n_traces
        self.assertEqual(len(params.params), number)
        self.assertEqual(params.params['fwhm_1'].value, fwhm)
        self.assertEqual(params.params['fwhm_1'].vary, opt_fwhm)
        self.assertEqual(params.params['fwhm_2'].expr, 'fwhm_1')

    @parameterized.expand([[0, True, 0.12, False, True],
                           [0, False, None, True, True],
                           [0, True, None, True, False],
                           [0.5, True, 0.18, False, False]])
    def test_adjustParams(self, t0, vary_t0, fwhm, opt_fwhm,
                          GVD_corrected):
        params = GlobalTargetParameters(n_traces, model)
        params.adjustParams(t0, vary_t0, fwhm, opt_fwhm,
                            GVD_corrected)
        self.assertEqual(params.params['t0_1'].value, t0)
        number = len(params_model) + 2*n_traces + exp_no*(n_traces + 1)
        if fwhm is not None:
            number += n_traces
            self.assertEqual(params.params['t0_1'].vary, vary_t0)
            self.assertEqual(params.params['fwhm_1'].vary, opt_fwhm)
            self.assertEqual(params.params['fwhm_1'].value, fwhm)
            self.assertEqual(params.params['fwhm_1'].value, fwhm)
            if GVD_corrected:
                self.assertEqual(params.params['t0_2'].expr, 't0_1')
            else:
                self.assertEqual(params.params['t0_2'].expr, None)
        else:
            # verify if there is no deconvolution t0 is fixed
            self.assertFalse(params.params['t0_1'].vary)
        self.assertEqual(len(params.params), number)


if __name__ == '__main__':
    unittest.main()
