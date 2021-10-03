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

#datasets_dir = "ultrafast/examples/dynamically_created_data/"

class TestDatasetsDAS(unittest.TestCase):
    """
    Generate exemplary datasets and fit them, to check if extracted
    parameters will reflect input of the dataset creator class.
    Dataset creator class is written in a different way compared to
    fitting routine, so there is small chance that some similar
    malfunction will occur in both generating/fitting code parts.
    """
    
    def setUp(self):
        self.datasets_dir = "../../examples/dynamically_created_data/"
    
    def skip_test_genAndFit3expNoConvNoNoiseDAS(self):
        #generate and save dataset, then fit it and verify results
        
        taus = [5,20,100]
        
        wave = DataSetCreator.generate_wavelength(400,700,31)
        peaks=[[470,550,650],[480,700],[500,600]]
        amplitudes=[[-10,-6,12],[-9,11],[-8,6]]
        fwhms=[[50,100,80],[40,70],[30,65]]
        shapes = DataSetCreator.generate_specific_shape(wave, taus=3, 
                                                        peaks=peaks, 
                                                        amplitudes=amplitudes, 
                                                        fwhms=fwhms)
        k1=1/taus[0]
        k2=1/taus[1]
        k3=1/taus[2]
        kmatrix = [[-k1,0,0],[0,-k2,0],[0,0,-k3]]
        initials = [0.33,0.34,0.33]
        profiles = DataSetCreator.generate_profiles(500.0,5000,
                                                    initials,kmatrix)
        data_set_conv = DataSetCreator.generate_dataset(shapes, 
                                                        profiles, 
                                                        0.2)
        new_times = DataSetCreator.generate_time(data_set_conv.index[0],
                                                 data_set_conv.index[-1]-0.01,
                                                 120)
        data_set_conv_proj = DataSetCreator.timegrid_projection(data_set_conv, 
                                                                new_times)        
        
        datapath = self.datasets_dir+"DAS_3exp_noconv_nonoise_test1.csv"
        
        data_set_conv_proj.to_csv(datapath)
        
        time, data, wavelength = read_data(datapath, wave_is_row = True)
        
        data_select, wave_select = select_traces(data, wavelength, 10)
        params = GlobExpParameters(data_select.shape[1], taus)
        params.adjustParams(0, vary_t0=False, vary_y0 = False, 
                            fwhm=0.2, opt_fwhm=True, vary_yinf=False)
        parameters = params.params
        
        fitter = GlobalFitExponential(time, data_select, 3, 
                                      parameters, True,
                                      wavelength=wave_select)
        
        fitter.allow_stop = False #in my case it just hangs.
        result = fitter.global_fit(maxfev=10000,
                                   use_jacobian = True,
                                   method='leastsq')
        
        explorer = ExploreResults(result)
        explorer.print_results()
        
        (x, data, wavelength, 
         params, exp_no, deconv, 
         tau_inf, svd_fit, type_fit, 
         derivative_space) = explorer._get_values()
        
        taus_out = []
        for i in range(len(taus)):
            taus_out.append(params["tau"+str(i+1)+"_1"].value)
        
        taus_err = []
        for i in range(len(taus)):
            taus_err.append(params["tau"+str(i+1)+"_1"].stderr)        
        
        for i in range(len(taus)):
            self.assertTrue(abs((taus[i]-taus_out[i])/taus_err[i]) < 5,
                            msg="""Tau generated is %.3f, tau after fit is 
                            %.3f, and error is %.3f""" % (taus[i],taus_out[i],taus_err[i]))   
        
    def skip_test_genAndFit1expNoConvNoNoiseDAS(self):
        #generate and save dataset, then fit it and verify results
        
        tau = 35.0
        
        wave = DataSetCreator.generate_wavelength(400,700,31)
        peaks=[[460,560,660],]
        amplitudes=[[-10,6,12],]
        fwhms=[[50,100,80],]
        shapes = DataSetCreator.generate_specific_shape(wave, taus=1, 
                                                        peaks=peaks, 
                                                        amplitudes=amplitudes, 
                                                        fwhms=fwhms)
        k1=1/tau
        kmatrix = [[-k1,],]
        initials = [1,]
        profiles = DataSetCreator.generate_profiles(300.0,5000,
                                                    initials,kmatrix)
        data_set_conv = DataSetCreator.generate_dataset(shapes, 
                                                        profiles, 
                                                        0.1)
        new_times = DataSetCreator.generate_time(data_set_conv.index[0],
                                                 data_set_conv.index[-1]-0.01,
                                                 120)
        data_set_conv_proj = DataSetCreator.timegrid_projection(data_set_conv, 
                                                                new_times)        
        
        datapath = self.datasets_dir+"DAS_1exp_noconv_nonoise_test1.csv"
        
        data_set_conv_proj.to_csv(datapath)
        
        time, data, wavelength = read_data(datapath, wave_is_row = True)
        
        data_select, wave_select = select_traces(data, wavelength, 300)
        params = GlobExpParameters(data_select.shape[1], [35,])
        params.adjustParams(0, vary_t0=False, vary_y0 = False, 
                            fwhm=0.1, opt_fwhm=True, vary_yinf=False)
        parameters = params.params
        
        fitter = GlobalFitExponential(time, data_select, 1, 
                                      parameters, True,
                                      wavelength=wave_select)
        
        fitter.allow_stop = False #in my case it just hangs.
        
        ##tests
        fitter._prepareJacobian(parameters)
        
        ##tests     
        
        result = fitter.global_fit(maxfev=10000, 
                                   #use_jacobian = True, 
                                   method='leastsq')
        
        explorer = ExploreResults(result)
        explorer.print_results()
        #explorer.plot_fit()
        #explorer.plot_DAS()  
        
        (x, data, wavelength, 
         params, exp_no, deconv, 
         tau_inf, svd_fit, type_fit, 
         derivative_space) = explorer._get_values()
        
        ##tests
        print(params)
        jac = fitter._jacobian(params)
        for i in range(jac.shape[0]):
            for j in range(jac.shape[1]):
                if(abs(jac[i,j]) > 10**(-1)):
                    print("Indexes %i, %i, value %.9f" % (i,j,jac[i,j]))
        
        print(jac.shape)
        ##tests
    
        tau_out = params["tau1_1"].value
        
        tau_err = params["tau1_1"].stderr
                  
        self.assertTrue(abs((tau-tau_out)/tau_err) < 5,
                        msg="""Tau generated is %.3f, tau after fit is 
                        %.3f, and error is %.3f""" % (tau,tau_out,tau_err))          
        
    def test_genAndTestJacobianAgainstNumerics(self):
        #technical test to check if written jacobian is correct agains numerics
        #warning! this test is very slow (not optimized), and worth running only
        #if some fundamental changes in core of the code happened...
        taus = [3,7,90]
        
        wave = DataSetCreator.generate_wavelength(400,700,31)
        peaks=[[470,550,650],[480,700],[500,600]]
        amplitudes=[[-10,-6,12],[-9,11],[-8,6]]
        fwhms=[[50,100,80],[40,70],[30,65]]
        shapes = DataSetCreator.generate_specific_shape(wave, taus=3, 
                                                        peaks=peaks, 
                                                        amplitudes=amplitudes, 
                                                        fwhms=fwhms)
        k1=1/taus[0]
        k2=1/taus[1]
        k3=1/taus[2]
        kmatrix = [[-k1,0,0],[0,-k2,0],[0,0,-k3]]
        initials = [0.33,0.34,0.33]
        profiles = DataSetCreator.generate_profiles(500.0,5000,
                                                    initials,kmatrix)
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
        
        time, data, wavelength = read_data(datapath, wave_is_row = True)
        
        data_select, wave_select = select_traces(data, wavelength, 1000)
        params = GlobExpParameters(data_select.shape[1], taus)
        params.adjustParams(0.1, vary_t0=True, vary_y0 = True, 
                            fwhm=0.2, opt_fwhm=True, vary_yinf=True)
        parameters = params.params
        
        self.tmp_fitter = GlobalFitExponential(time, data_select, 3, 
                                      parameters, True,
                                      wavelength=wave_select)
        
        params = self.tmp_fitter.params
        
        print(params)
        
        self.tmp_fitter._prepareJacobian(params)
        params_no = len(self.tmp_fitter.recent_key_array)
 
        ndata, nx = self.tmp_fitter.data.shape #(no of taus,no of lambdas)
        out_funcs_no = nx*self.tmp_fitter.x.shape[0] #no of residuals
       
        self.real_param_num = 0
        for par_i in range(params_no):
            if(self.tmp_fitter.recent_constrainted_array[par_i] is True):
                continue #skip constrained or fixed params
            
            self.checkkey = self.tmp_fitter.recent_key_array[par_i]
            
            for res_index in range(out_funcs_no):
                self.checkres = res_index
                self.params_tmp1 = params.copy()
                self.params_tmp2 = params.copy()
                
                par_val = params[self.checkkey].value
                num_grad = self.numericGradient(par_val, epsilon = 1.49e-08)
                anal_grad = self.gradientFunc(par_val)
                diff = num_grad - anal_grad
                
                if(abs(diff) > 10**(-8)):
                    print(("Num gives %.9f anal gives %.9f " % (num_grad,anal_grad)) +\
                          str(self.checkkey)+" and res number "+str(res_index))

                
            self.real_param_num += 1
            

    def numericGradient(self, value, epsilon = 1.49e-08):
        return (self.objectiveFunc(value+epsilon/2) - \
                self.objectiveFunc(value-epsilon/2))/epsilon
        
    def objectiveFunc(self, x):
        self.params_tmp1[self.checkkey].value = x
        restmp = self.tmp_fitter._objective(self.params_tmp1)
        return restmp[self.checkres]

    def gradientFunc(self, x):
        self.params_tmp2[self.checkkey].value = x
        jactmp = self.tmp_fitter._jacobian(self.params_tmp2)
        return jactmp[self.real_param_num,self.checkres]
     
        
          
if __name__ == '__main__':
    unittest.main()