# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:48:51 2020

@author: 79344
"""
from lmfit import Parameters

class GlobExpParameters():
    def __init__(self,number_traces,taus):
        '''Class parameterd generator
        atributes
        ----------
        taus: list
            contains the value of the decay times
            
        exp_no: int 
            is the number of exponentials and equal to the len of taus
        
        number_traces: int
            contains the number of data sets that should be globally fitted
        
        params: lmFit Parameters class 
            contains the parameters for the fit
            '''
        
        self.taus=[taus] if type(taus) == float or type(taus)== int else taus
        self.exp_no=len(self.taus)
        self.number_traces=number_traces
        self.params=Parameters()
        
    def _generateParams(self,t0,vary_t0):
        ''' generate the parameters for globally fitting the number of traces 
        to a sum of "exp_no" exponential decay'''
        for iy in range(self.number_traces):
            self.params.add_many(('y0_' +str(iy+1), 0,True, None, None, None, None),
                            ('t0_'+str(iy+1), t0, vary_t0,  0, None, None, None))
            
            for i in range (self.exp_no):
                # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                self.params.add_many(('pre_exp%i_' %(i+1) +str (iy+1), 0.1*10**(-i), True, None, None, None, None),
                                ('tau%i_' %(i+1) +str (iy+1), self.taus[i], True, 0.00000001, None, None, None))

    def _addDeconvolution(self, fwhm,opt_fwhm,tau_inf=1E12):
        '''add the deconvolution parameters to a sum of "exp_no" exponential decay'''
        for iy in range(self.number_traces):
            self.params.add_many(('fwhm_' +str(iy+1), fwhm, opt_fwhm, 0.000001, None, None, None))
            if tau_inf is not None:            
                self.params.add_many(('yinf_' +str (iy+1), 0.001, True, None, None, None, None))
    
    def adjustParams(self,t0,vary_t0=True,fwhm=0.12,opt_fwhm=False,GVD_corrected=True,tau_inf=1E12):
        """function to initialize parameters
        
        Paramaters
        ----------
        t0: int or float
            the t0 for the fitting 
            
        vary_t0: bool
            default True, allows to optimize t0 when the sum of exponential is convolve with a gaussian 
            If there is no deconvolution t0 is always fix
        
        fwhm: float or None
            default 0.12. FWHM of the the laser pulse use in the experiment
            If None. the deconvolution parameters will not be added
        
        opt_fwhm: bool
            default: False. allows to optimized the FWHM. 
            Thoretically this should be meassured externally and be fix
        
        GVD_corrected: bool
            default True. If True all traces will have same t0. If False t0 is independent of the trace
        
        tau_inf: int or float:
            allows to add a constant decay value to the parameters.
            This modellized photoproducts formation with long decay times
            If None tau_inf is not added.
        """
        self._generateParams(t0,vary_t0)
        for iy in range(2,self.number_traces+1):
            self.params['t0_%i' % iy].expr='t0_1'
            for i in range (self.exp_no):
                self.params['tau%i_' %(i+1) +str (iy)].expr='tau%i_1' %(i+1)
        if fwhm  is not None:
            self._addDeconvolution(fwhm,opt_fwhm,tau_inf=tau_inf)
            if GVD_corrected == False:
                for iy in range(2,self.number_traces+1):
                    self.params['t0_%i' % iy].expr=None
                    self.params['t0_%i' % iy].vary=True
        else:
            self.params['t0_1'].vary=False