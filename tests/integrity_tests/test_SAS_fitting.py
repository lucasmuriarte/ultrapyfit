#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:44:42 2021

@author: staszek
"""


import unittest
from ultrafast.utils.divers import DataSetCreator
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.experiment import Experiment
from ultrafast.fit.targetmodel import ModelWindow, Model
import numpy as np
import copy


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
        self.models_dir = "../../examples/target_models/"
    
    def test_genAndFit3expNoNoiseSequential(self):
        #generate and save dataset, then fit it and verify results
        
        taus = np.array([5,20,100])
        ks = 1/taus
        
        wave = DataSetCreator.generate_wavelength(400,700,31)
        peaks=[[470,550,650],[480,700],[500,600]]
        amplitudes=[[-10,-6,12],[-9,11],[-8,6]]
        fwhms=[[50,100,80],[40,70],[30,65]]
        shapes = DataSetCreator.generate_specific_shape(wave, taus=3, 
                                                        peaks=peaks, 
                                                        amplitudes=amplitudes, 
                                                        fwhms=fwhms)

        kmatrix = [[-ks[0],0,0],[ks[0],-ks[1],0],[0,ks[1],-ks[2]]]
        initials = [1,0.0,0.0]
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
        
        datapath = self.datasets_dir+"SAS_3exp_nonoise_sequential_test1.csv"
        
        data_set_conv_proj.to_csv(datapath)
        
        experiment = Experiment.load_data(datapath, wave_is_row=True)
        
        experiment.select_traces(points='all')
        
        #experiment.fitting.initialize_target_model_window()
        
        testmodel = Model()
        testmodel = Model.load(self.models_dir+"3exp_sequential.model")
        #testmodel.manualModelBuild_V2()
        #testmodel.save(self.models_dir+"3exp_sequential.model")
        #testmodel.manualModelBuild_V2()
        
        testmodel.genParameters()
        experiment.fitting.fit_records.target_models[1] = testmodel
        experiment.fitting._model_params = testmodel.params
        model_names = experiment.fitting._model_params.model_names
        experiment.fitting._experiment._add_action(f'Target model loaded')
        #it would be nice to have method like "take out Model object" and
        #"put in Model object", so one could independently manage some zoo
        #of models and automatize loading into fitting procedure
        
        experiment.fitting.initialize_target_params(0, 0.20, model=1)
        
        #i had to add this, completely don't undrestand why...
        experiment.fitting._model_params.model_names = model_names
        
        experiment.fitting.fit_global()

        #print(experiment.fitting.fit_records.global_fits[1].params)

        ks_out = []
        for i in range(len(ks)):
            ks_out.append(experiment.fitting.fit_records.global_fits[1].params["k_"+str(i+1)+str(i+1)].value)
        
        ks_err = []
        for i in range(len(ks)):
            #ks_err.append(experiment.fitting.fit_records.global_fits[1].params["k_"+str(i+1)+str(i+1)].stderr)  
            ks_err.append(np.abs(experiment.fitting.fit_records.global_fits[1].params["k_"+str(i+1)+str(i+1)].value/20)) 
            #assumed 5% error
        #why there are no errors! in DAS they were there!
        
        print(ks_err)
        
        #test equality of taus and preexps within predefined range (because no errors available)
        for i in range(len(taus)):
            self.assertTrue(abs((ks[i]-ks_out[i])/ks_err[i]) < 5,
                            msg="""Tau generated is %.3f, tau after fit is 
                            %.3f, and error is %.3f""" % (ks[i],ks_out[i],ks_err[i]))       
            
            for i_wave in range(len(wave)):
                preexp = shapes.iloc[i,i_wave]
                preexp_err = experiment.fitting.fit_records.global_fits[1].params["pre_exp"+str(i+1)+"_"+str(i_wave+1)].stderr
                preexp_out = experiment.fitting.fit_records.global_fits[1].params["pre_exp"+str(i+1)+"_"+str(i_wave+1)].value            
                
                gen_preexps_list = [shapes.iloc[i_t,i_wave] for i_t in range(len(taus))]
                if(np.abs(preexp) < sum(np.abs(gen_preexps_list))/10):
                    continue #just skip places where given component had no bands
                
                self.assertTrue(abs((preexp-preexp_out)/preexp_err) < 5,
                                msg="""Preexp[%i,%i] generated is %.6f, preexp after fit is 
                                %.6f, and error is %.6f""" % (i,i_wave,preexp,preexp_out,preexp_err))  



if __name__ == '__main__':
    unittest.main()