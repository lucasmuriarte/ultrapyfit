# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:10:17 2020

@author: 79344
"""

import unittest
import numpy as np
from chempyspec.ultrafast.GlobExpParams import GlobExpParameters
from parameterized import parameterized
import sys


taus=[8,30,200]
number_traces=5


class TestBasicSpectrum(unittest.TestCase):
    '''test for GlobExpParameters Class'''
    
    @parameterized.expand([[0, False],[5,True]])
    def test__generateParams(self,t0,vary):
        params=GlobExpParameters(number_traces,taus)
        params._generateParams(t0,vary)
        #two parameter per tau including t0 thus: len(taus)+1
        number=(len(taus)+1)*2*number_traces
        self.assertEqual(len(params.params),number)
        self.assertEqual(params.params['tau1_1'].value,taus[0])
        self.assertEqual(params.params['tau2_1'].value,taus[1])
        self.assertEqual(params.params['tau3_1'].value,taus[2])
        self.assertEqual(params.params['t0_1'].value,t0)
        self.assertEqual(params.params['t0_1'].vary,vary)
        
        
    @parameterized.expand([[0.12,False,1E12],
                           [0.18,True,None],
                           [0.15,False,1E8]])
    def test__addDeconvolution(self,fwhm,opt_fwhm,tau_inf):
        params=GlobExpParameters(number_traces,taus)
        params._generateParams(0,False)
        params._add_deconvolution(fwhm, opt_fwhm, tau_inf)
        #notice tau_inf is not a parameter and thus is not added only its preexponential function plus fwhm
        number = (len(taus)+2)*2*number_traces 
        if tau_inf is None: number -= number_traces
        self.assertEqual(len(params.params),number)
        self.assertEqual(params.params['fwhm_1'].value,fwhm)
        self.assertEqual(params.params['fwhm_1'].vary,opt_fwhm)

   
    @parameterized.expand([[0,True,0.12,False,True,1E12],
                           [0,False,None,True,True,None],
                           [0,True,None,True,False,None],
                           [0.5,True,0.18,False,False,1E12]])      
    def test_adjustParams(self,t0,vary_t0,fwhm,opt_fwhm,GVD_corrected,tau_inf):
        params=GlobExpParameters(number_traces,taus)
        params.adjustParams(t0,vary_t0,fwhm,opt_fwhm,GVD_corrected,tau_inf)
        self.assertEqual(params.params['tau1_1'].value,taus[0])
        self.assertEqual(params.params['tau2_1'].value,taus[1])
        self.assertEqual(params.params['tau3_1'].value,taus[2])
        self.assertEqual(params.params['tau1_2'].expr,'tau1_1')
        self.assertEqual(params.params['tau2_2'].expr,'tau2_1')
        self.assertEqual(params.params['tau3_2'].expr,'tau3_1')
        self.assertEqual(params.params['t0_1'].value,t0)
        number=(len(taus)+1)*2*number_traces
        if fwhm is not None:
            self.assertEqual(params.params['t0_1'].vary,vary_t0)
            number = number + number_traces*2 if  tau_inf is not None else  number + number_traces
            self.assertEqual(params.params['fwhm_1'].vary,opt_fwhm)
            self.assertEqual(params.params['fwhm_1'].value,fwhm)
            self.assertEqual(params.params['fwhm_1'].value,fwhm)
            if GVD_corrected:
                self.assertEqual(params.params['t0_2'].expr,'t0_1')
            else:
                self.assertEqual(params.params['t0_2'].expr,None)
        else:
            #verify if there is no deconvolution t0 is fixed
            self.assertFalse(params.params['t0_1'].vary)
        self.assertEqual(len(params.params),number)
        
if __name__ == '__main__':
    unittest.main()    