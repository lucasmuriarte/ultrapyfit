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
    
    def test_genAndFit3expNoConvNoNoiseDAS(self):
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
                            fwhm=0.2, opt_fwhm=False, vary_yinf=False)
        parameters = params.params
        
        fitter = GlobalFitExponential(time, data_select, 3, 
                                      parameters, True,
                                      wavelength=wave_select)
        
        fitter.allow_stop = False #in my case it just hangs.
        result = fitter.global_fit(maxfev=10000,
                                   use_jacobian = True,
                                   method='leastsq')
        
        explorer = ExploreResults(result)
        #explorer.print_results()
        
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
        
    def test_genAndFit1expNoConvNoNoiseDAS(self):
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

        result = fitter.global_fit(maxfev=10000, 
                                   use_jacobian = True, 
                                   method='leastsq')
        
        explorer = ExploreResults(result)
        #explorer.print_results()
        #explorer.plot_fit()
        #explorer.plot_DAS()  
        
        (x, data, wavelength, 
         params, exp_no, deconv, 
         tau_inf, svd_fit, type_fit, 
         derivative_space) = explorer._get_values()
    
        tau_out = params["tau1_1"].value
        tau_err = params["tau1_1"].stderr
                  
        self.assertTrue(abs((tau-tau_out)/tau_err) < 5,
                        msg="""Tau generated is %.3f, tau after fit is 
                        %.3f, and error is %.3f""" % (tau,tau_out,tau_err))          
        

if __name__ == '__main__':
    unittest.main()