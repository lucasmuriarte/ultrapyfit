# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:55:03 2020

@author: lucas
"""
import numpy as np
from ExploreResultsClass import ExploreResults
from PlotTransientClass import ExploreData
from GlobExpParams import GlobExpParameters
from outils import define_weights, UnvariableContainer, LabBook
from PreprocessingClass import Preprocessing
from GlobalFitClass import GlobalFitExponential
from GlobalTargetClass import GlobalFitTargetModel
import os

class ExperimentException(Exception):
    """General Purpose Exception."""
    
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return "{}".format(self.msg)

class Experiment(ExploreData,ExploreResults):
    
    def __init__(self,x, data, wavelength=None):
        if wavelength is None:
            self.wavelength=wavelength
            general_cal='\tNone'
        else:
            self.wavelength=np.array(wavelength)
            general_cal='\tAlready calibrated'
        self.x=x
        self.data=data
        self.selected_traces = None
        self.selected_wavelength = None
        self.data_sets=UnvariableContainer()
        self.data_sets.original_data={'time':x,'data':data,'wavelength':wavelength}
        self.excitation=None
        self._fit_number=0
        self.units={'time_unit':'ps','time_unit_high':'ns','time_unit_low':'fs',
                    'wavelength_unit':'nm','factor_high':1000,'factor_low':1000}
        self._params_initialized=False
        self.GVD_corrected=False
        self.deconv=True
        self.SVD_fit=False
        self.type_fit='Exponential'
        self.fit_records=LabBook()
        self.fit_records.single_fits={}
        self.fit_records.bootstrap_record={}
        self.fit_records.conf_interval={}
        self.fit_records.target_models={}
        self.fit_records.global_fits={}
        self.params=None
        self.weights={'apply':False,'vector':None,'range':[],'type':'constant','value':2}
        self.cmap='viridis'
        self.general_report={'File':'\tNone','Excitation':self.excitation,
                             'Units':{'Time unit':'\tps','Wavelength unit':'nm'},
                             'Data Shape':{'Initial number of traces':f'{data.shape[1]}','Initial time points':
                                         f'{data.shape[0]}','Actual number of traces':'All','Actual time points':'All'},
                             'Preprocessing':{'Calibration':general_cal,'GVD correction':[],'IRF Fit':'\tNone','Baseline correction':None,
                                              'Cutted Wavelengths':[],'Cutted Times':[],'Deleted wavelength points':[],
                                              'Deleted time points':[],'Average time points':[],'Time shift':[],
                                              'Polynom fit':[],'Derivate data':False},
                             'Fits done':{},'Sequence of actions':[],'User comments':[]}
        ExploreData.__init__(self,self.x,self.data,self.wavelength,self.selected_traces,self.selected_wavelength,self.cmap)
        ExploreResults.__init__(self,self.fit_records.global_fits)
    
    @property
    def chirp_corrected(self):
        return self.GVD_corrected
    
    @chirp_corrected.setter
    def chirp_corrected(self,value):
        self.GVD_corrected = value
    
    def _addToGeneralReport(self,key,val,sub_key=None):
        '''modify general report dictionary'''
        if key == 'Sequence of actions':
            self.general_report['Sequence of actions'].append(val)
        elif sub_key is not None:
            self.general_report[key][sub_key]=val
        else:
            self.general_report[key]=val
        
    def defineUnits(self,time,wavelength):
        times = ['Ato s','fs','ps','ns','μs','ms','s','min','h']
        assert (time in times[1:-1]) or time=='us'
        assert type(wavelength) == str
        if time == 'us':
            self.units['time_unit'] = 'µs'
            self.units['time_unit_high'] = 'ms'
            self.units['time_unit_low'] = 'ns'
        else:
            index=times.index(time)
            self.units['time_unit'] = time
            self.units['time_unit_high'] = times[index+1]
            self.units['time_unit_low'] = times[index-1]
        if self.units['time_unit'] == 's':
            self.units['factor_high'] = 60
        elif self.units['time_unit'] == 'min':
            self.units['factor_high'] = 60
            self.units['factor_low'] = 60
        else:
            pass
        self.units['wavelength_unit']=wavelength
        
        #general report changes
        self._addToGeneralReport('Units',f'\t{self.time_unit}','Time unit')
        self._addToGeneralReport('Units',self.units['wavelength_unit'],'Wavelength unit')
        self._addToGeneralReport('Sequence of actions','\t--> Units changed')
                         
    def baselineSubstraction(self, nuber_spec=2,only_one=False):
        self.data_sets.before_baseline_substraction=\
        {'time':self.x,'data':self.data,'wavelength':self.wavelength}
        new_data=Preprocessing.baselineSubstraction(self.data,nuber_spec=2,only_one=False)
        self.data=new_data
        if only_one:
            string = f'Substracted time {nuber_spec} spectrum'
        else:
            init,final=0,nuber_spec if type(nuber_spec) == int else nuber_spec[0],nuber_spec[1]
            string = f'Substracted average {init}-{final} spectra'
        
        #general report changes
        self._addToGeneralReport('Preprocessing',string,'Baseline correction')
        self._addToGeneralReport('Sequence of actions','\t--> Baseline Substraction')
    
    def cutTime(self, mini=None,maxi=None):
        self.data_sets.before_cut_time=\
        {'time':self.x,'data':self.data,'wavelength':self.wavelength}
        new_data,new_x=Preprocessing.cutTime(self.data,self.x,mini,maxi)
        self.data,self.x=new_data,new_x
        
        #general report changes
        units=self.units["time_unit"]
        mini_str = f'from {mini} {units}'if mini is not None else ''
        maxi_str = f'until {maxi} {units}'if maxi is not None else ''
        action=f'\t\tSelected data {mini_str} {maxi_str}'
        self.general_report['Preprocessing']['Cutted Times'].append(action)
        self._addToGeneralReport('Sequence of actions',f'\t--> Cut or selection of time range')
        self._addToGeneralReport('Data Shape',self.data.shape[0],'Actual time points')
    
    def averageTimePoints(self, starting_point, step, method='log',grid_dense=5):
        self.data_sets.before_average_time=\
        {'time':self.x,'data':self.data,'wavelength':self.wavelength}
        new_data,new_x=Preprocessing.averageTimePoints(self.data,self.x,starting_point, step, method, grid_dense)
        self.data,self.x=new_data,new_x
        
        #general report changes
        self._addToGeneralReport('Preprocessing',f'\t\tAverage from {starting_point}, with {method} {step} step','Average time points')
        self._addToGeneralReport('Sequence of actions','\t--> Average of time points')
        self._addToGeneralReport('Data Shape',self.data.shape[0],'Actual time points')
        
    def derivateData(self, window_length=25,polyorder=3,deriv=1,mode='mirror'):
        self.data_sets.before_derivation=\
        {'time':self.x,'data':self.data,'wavelength':self.wavelength}
        new_data=Preprocessing.derivateData(window_length,polyorder,deriv,mode)
        self.data=new_data
        
        #general report changes
        string ='\t'+'\t\t\t'.join([f'{key}: {self.derivative_space[key]}\n' for key in self.derivative_space])
        self._addToGeneralReport('Preprocessing',string,'Derivate data')
        self._addToGeneralReport('Sequence of actions','\t--> Derivation of Data')

    def cutWavelenghts(self, left=None,right=None,innercut=False):    
        self.data_sets.before_cut_wavelenght=\
        {'time':self.x,'data':self.data,'wavelength':self.wavelength}    
        new_data,new_wave=Preprocessing.cutWavelenghts(left,right,innercut)
        self.data,self.wavelength=new_data,new_wave
        
        #general report changes
        units=self.units["wavelength_unit"]
        left_str = f'from {left} {units}'if left is not None else ''
        right_str = f'until {right} {units}'if right is not None else ''
        mode = 'Cutted' if innercut else 'Selected'
        action=f'\t\t{mode} data {left_str} {right_str}'
        self.general_report['Preprocessing']['Cutted Wavelengths'].append(action)
        self._addToGeneralReport('Sequence of actions',f'\t--> Cut or selection of wavelength')
        self._addToGeneralReport('Data Shape',self.data.shape[1],'Actual number of traces')
        
    def delPoints(self, points,dimension='time'):
        self.data_sets.before_delete_point=\
        {'time':self.x,'data':self.data,'wavelength':self.wavelength}    
        if dimension == 'time':
            new_data,new_x=Preprocessing.delPoints(points,self.data,self.x)
            self.data,self.x=new_data,new_x
        elif dimension == 'wavelength':
            new_data,new_wave=Preprocessing.delPoints(points,self.data,self.wavelength)
        else:
            raise ExperimentException('dimension should be "time" or "wavelength"')
        
        #general report changes
        lista=self.general_report['Preprocessing']['Deleted {dimension} points'] + points
        self._addToGeneralReport('Preprocessing',lista,'Deleted {dimension} points')
        self._addToGeneralReport('Data Shape',self.data.shape[0],'Actual number of traces')
        self._addToGeneralReport('Data Shape',self.data.shape[1],'Actual time points')

    def shitTime(self,value):
        self.x=self.x-value
    
    def defineWeights(self,rango,typo='constant',val=5):
        self.weights=define_weights(self.x, rango, typo='constant', val=5)

    def createNewDir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.working_directory=path
        self.save['path']=self.working_directory
    
    def initializeExpParams(self,t0,*taus,fwhm=0.12,tau_inf=1E12,opt_fwhm=False):
        taus = list(taus)
        param_creator = GlobExpParameters(self.selected_traces,taus)
        if fwhm == None: 
            self._deconv=False
            vary_t0=False
        else :
            vary_t0=True
            self._deconv=True
            self.tau_inf=tau_inf
        param_creator.adjustParams(t0,vary_t0,fwhm,opt_fwhm,self.GVD_corrected,tau_inf)
        self.params = param_creator.params
        self._params_initialized = 'Global'
        
        #general report changes
        self._addToGeneralReport('Sequence of actions','\t--> New parameters initialized')
        
    def finalFit(self,vary=True,maxfev=5000,apply_weights=False):
        if self._params_initialized == 'Global':
            minimizer=GlobalFitExponential(self.selected_traces, self.selected_wavelength, self.params, \
                                           self._deconv, self.tau_inf, GVD_corrected=self.GVD_corrected)
        elif self._params_initialized == 'Target':
            minimizer=GlobalFitTargetModel(self.selected_traces,self.selected_wavelength,self.params)
        else:
             raise ExperimentException('Parameters need to be initiliazed first"')
        if apply_weights:
            minimizer.weights = self.weights
        results = minimizer.finalFit(vary,maxfev,apply_weights)
        results.details['SVD_fit']=self.SVD_fit
        self.fit_records.global_fits[self._fit_number]=results
        self._fit_number += 1
        
        
        
        
        
        
        
        