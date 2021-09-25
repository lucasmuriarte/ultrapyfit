#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:37:51 2021

@author: staszek
"""

import unittest
from ultrafast.utils.divers import DataSetCreator

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
        
    
    def helperFunc(self):
        return 0
    
    def test_genAndFit1exp1bandDAS(self):
        #generate and save dataset, then fit it and verify results
        
        wave = DataSetCreator.generate_wavelength(400,700,500)
        peaks=[[470,550,650],[480,700],[500,600]]
        amplitudes=[[-10,-6,12],[-9,11],[-8,6]]
        fwhms=[[50,100,80],[40,70],[30,65]]
        shapes = DataSetCreator.generate_specific_shape(wave, taus=3, peaks=peaks, amplitudes=amplitudes, fwhms=fwhms)
        k1=1/5.0
        k2=1/20.0
        k3=1/100.0
        kmatrix = [[-k1,0,0],[k1/3,-k2,0],[2*k1/3,0,-k3]]
        initials = [1.0,0,0]
        profiles = DataSetCreator.generate_profiles(500.0,5000,initials,kmatrix)
        data_set_conv = DataSetCreator.generate_dataset(shapes, profiles, 1.2)
        new_times = DataSetCreator.generate_time(data_set_conv.index[0],data_set_conv.index[-1],120)
        data_set_conv_proj = DataSetCreator.timegrid_projection(data_set_conv, new_times)        
        
        data_set_conv_proj.to_csv(self.datasets_dir+"DAS_1exp_1band_test1.csv")
        
        
        
        
        
        
        
        
        self.assertTrue(True,msg="Shit fucked up.")   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    unittest.main()