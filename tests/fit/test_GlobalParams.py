# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:10:17 2020

@author: 79344
"""

import unittest, os
from ultrafast.fit.GlobalParams import GlobExpParameters, GlobalTargetParameters
from parameterized import parameterized
from ultrafast.fit.targetmodel import Model
from ultrafast.utils.divers import get_root_directory

class TestGlobExpParameters(unittest.TestCase):
    """
    test for GlobExpParameters Class
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.taus = [8, 30, 200]
        cls.n_traces = 5
        cls.model = Model.load(
            os.path.join(get_root_directory(), "tests/fit/testmodel2.model"))
        cls.params_model = cls.model.genParameters()
        exp_no = cls.params_model['exp_no']
    
    @parameterized.expand([[0, False], [5, True]])
    def test_generateParams(self, t0, vary):
        params = GlobExpParameters(self.n_traces, self.taus)
        params._generateParams(t0, vary)
        # two parameter per tau including t0 thus: len(taus)+1
        number = (len(self.taus)+1) * 2 * self.n_traces
        self.assertEqual(len(params.params), number)
        self.assertEqual(params.params['tau1_1'].value, self.taus[0])
        self.assertEqual(params.params['tau2_1'].value, self.taus[1])
        self.assertEqual(params.params['tau3_1'].value, self.taus[2])
        self.assertEqual(params.params['t0_1'].value, t0)
        self.assertEqual(params.params['t0_1'].vary, vary)

    @parameterized.expand([[0.12, False, 1E12],
                           [0.18, True, None],
                           [0.15, False, 1E8]])
    def test_addDeconvolution(self, fwhm, opt_fwhm, tau_inf):
        params = GlobExpParameters(self.n_traces, self.taus)
        params._generateParams(0, False)
        params._add_deconvolution(fwhm, opt_fwhm, tau_inf)
        # notice tau_inf is not a parameter and thus is not
        # added only its preexponential function plus fwhm
        number = (len(self.taus)+2) * 2 * self.n_traces
        if tau_inf is None:
            number -= self.n_traces
        self.assertEqual(len(params.params), number)
        self.assertEqual(params.params['fwhm_1'].value, fwhm)
        self.assertEqual(params.params['fwhm_1'].vary, opt_fwhm)
        self.assertEqual(params.params['fwhm_2'].expr, 'fwhm_1')

    @parameterized.expand([[0, True, 0.12, False, True, 1E12, None],
                           [0, False, None, True, True, None, 0],
                           [0, True, None, True, False, None, 5],
                           [0.5, True, 0.18, False, False, 1E12, 10]])
    def test_adjustParams(self, t0, vary_t0, fwhm, opt_fwhm,
                          GVD_corrected, tau_inf, y0):
        params = GlobExpParameters(self.n_traces, self.taus)
        params.adjustParams(t0, vary_t0, fwhm, opt_fwhm, 
                            GVD_corrected, tau_inf)
        self.assertEqual(params.params['tau1_1'].value, self.taus[0])
        self.assertEqual(params.params['tau2_1'].value, self.taus[1])
        self.assertEqual(params.params['tau3_1'].value, self.taus[2])
        self.assertEqual(params.params['tau1_2'].expr, 'tau1_1')
        self.assertEqual(params.params['tau2_2'].expr, 'tau2_1')
        self.assertEqual(params.params['tau3_2'].expr, 'tau3_1')
        self.assertEqual(params.params['t0_1'].value, t0)
        number = (len(self.taus)+1) * 2 * self.n_traces
        if fwhm is not None:
            self.assertEqual(params.params['t0_1'].vary, vary_t0)
            if tau_inf is not None:
                number = number + self.n_traces * 2
            else:
                number + self.n_traces
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

class TestGlobalTargetParams(TestGlobExpParameters):
    def test__init__(self):
        params = GlobalTargetParameters(self.n_traces, self.model)
        self.assertTrue(params.params == self.params_model)

    @parameterized.expand([[0, False, False],
                           [5, True, True],
                           [5, True, False],
                           [5, False, True]])
    def test_add_preexp_t0_y0(self, t0, vary, gvd_corrected):
        params = GlobalTargetParameters(self.n_traces, self.model)
        params._add_preexp_t0_y0(t0, vary, gvd_corrected)
        # two parameter per tau including t0 thus: len(taus)+1
        number = len(self.params_model) + 2 * self.n_traces + self.exp_no * (self.n_traces + 1)
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
    def test_addDeconvolution(self, fwhm, opt_fwhm):
        params = GlobalTargetParameters(self.n_traces, self.model)
        params._add_deconvolution(fwhm, opt_fwhm)
        # notice tau_inf is not a parameter and thus is not
        # added only its preexponential function plus fwhm
        number = len(self.params_model) + self.n_traces
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
        params = GlobalTargetParameters(self.n_traces, self.model)
        params.adjustParams(t0, vary_t0, fwhm, opt_fwhm,
                            GVD_corrected)
        self.assertEqual(params.params['t0_1'].value, t0)
        number = len(self.params_model) + 2*self.n_traces + self.exp_no*(self.n_traces + 1)
        if fwhm is not None:
            number += self.n_traces
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
