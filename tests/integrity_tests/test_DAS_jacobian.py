#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:37:51 2021

@author: staszek
"""

import unittest
from ultrafast.utils.divers import DataSetCreator
from ultrafast.fit.GlobalFit import GlobalFitExponential
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.fit.GlobalParams import GlobExpParameters
from scipy.optimize import check_grad, approx_fprime
import numpy as np

# datasets_dir = "ultrafast/examples/dynamically_created_data/"


class TestDatasetsDAS(unittest.TestCase):
    """
    Generate exemplary dataset and check if jacobian is correctly implemented.
    """
    
    def setUp(self):
        self.datasets_dir = "../../examples/dynamically_created_data/"         
        
    def test_genAndTestJacobianAgainstNumerics(self):
        # technical test to check if written jacobian is correct agains numerics
        # warning! this test is very slow (not optimized), and worth running
        # only if some fundamental changes in core of the code happened...
        taus = [3, 7, 90]
        
        wave = DataSetCreator.generate_wavelength(400, 700, 31)
        peaks = [[470, 550, 650], [480, 700], [500, 600]]
        amplitudes = [[-10, -6, 12], [-9, 11], [-8, 6]]
        fwhms = [[50, 100, 80], [40, 70], [30, 65]]
        shapes = DataSetCreator.generate_specific_shape(wave, taus=3, 
                                                        peaks=peaks, 
                                                        amplitudes=amplitudes, 
                                                        fwhms=fwhms)
        k1 = 1/taus[0]
        k2 = 1/taus[1]
        k3 = 1/taus[2]
        kmatrix = [[-k1, 0, 0], [0, -k2, 0], [0, 0, -k3]]
        initials = [0.33, 0.34, 0.33]
        profiles = DataSetCreator.generate_profiles(500.0, 5000,
                                                    initials, kmatrix)
        data_set_conv = DataSetCreator.generate_dataset(shapes, 
                                                        profiles, 
                                                        0.2)
        new_times = DataSetCreator.generate_time(data_set_conv.index[0],
                                                 data_set_conv.index[-1]-0.01,
                                                 50)
        data_set_conv_proj = DataSetCreator.timegrid_projection(data_set_conv, 
                                                                new_times)        
        
        datapath = self.datasets_dir+"DAS_3exp_noconv_nonoise_test_j.csv"
        
        data_set_conv_proj.to_csv(datapath)
        
        time, data, wavelength = read_data(datapath, wave_is_row=True)
        
        data_select, wave_select = select_traces(data, wavelength, 50)
        params = GlobExpParameters(data_select.shape[1], taus)
        params.adjustParams(0.0, vary_t0=True, fwhm=0.2, opt_fwhm=True)
        parameters = params.params
        
        self.tmp_fitter = GlobalFitExponential(time, data_select, 3, 
                                               parameters, True,
                                               wavelength=wave_select)
        
        params = self.tmp_fitter.params
        
        # print(params) #diagnostics
        
        self.tmp_fitter._prepare_jacobian(params)
        params_no = len(self.tmp_fitter.recent_key_array)
 
        ndata, nx = self.tmp_fitter.data.shape  # (no of taus,no of lambdas)
        out_funcs_no = nx*self.tmp_fitter.x.shape[0]  # no of residuals
       
        self.real_param_num = 0
        for par_i in range(params_no):
            if self.tmp_fitter.recent_constrainted_array[par_i] is True:
                continue  # skip constrained or fixed params
            
            self.checkkey = self.tmp_fitter.recent_key_array[par_i]
            
            for res_index in range(out_funcs_no):
                self.checkres = res_index
                self.params_tmp1 = params.copy()
                self.params_tmp2 = params.copy()
                
                par_val = params[self.checkkey].value
                num_grad = self.numericGradient(par_val)
                anal_grad = self.gradientFunc(par_val)
                diff = num_grad - anal_grad
                
                # diagnostics
                # if(abs(diff) > 10**(-8)):
                #     print(("Num gives %.9f anal gives %.9f " % (num_grad,anal_grad)) +\
                #           str(self.checkkey)+
                #           " and res number "+str(res_index))
                msg = "Num gives %.9f anal gives %.9f key " % (num_grad,
                                                               anal_grad)
                msg = msg + str(self.checkkey) + " and res number " + \
                      str(res_index)

                self.assertTrue(abs(diff) < 10**(-8), msg=msg)
            self.real_param_num += 1

    def numericGradient(self, value, epsilon=1.49e-08):
        return (self.objectiveFunc(value+epsilon/2) -
                self.objectiveFunc(value-epsilon/2))/epsilon
        
    def objectiveFunc(self, x):
        self.params_tmp1[self.checkkey].value = x
        restmp = self.tmp_fitter._objective(self.params_tmp1)
        return restmp[self.checkres]

    def gradientFunc(self, x):
        self.params_tmp2[self.checkkey].value = x
        jactmp = self.tmp_fitter._jacobian(self.params_tmp2)
        return jactmp[self.real_param_num, self.checkres]

          
if __name__ == '__main__':
    unittest.main()
