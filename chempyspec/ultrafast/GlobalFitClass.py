# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:00:23 2020

@author: lucas
"""
import numpy as np
import lmfit
from ModelCreatorClass import  ModelCreator
from  GlobExpParams import GlobExpParameters

def globalFitSumExponetial(x, data, *taus, vary=True, t0=0,  maxfev=5000, **kwargs):
    taus=list(taus)
    if type(vary) == bool:
        vary = [vary for i in taus]
    exp_no=len(taus)
    _ , n_traces = data.shape
    params=GlobExpParameters(n_traces, taus)
    params.adjustParams(t0,False)
    fit=GlobalFitExponetial(x, data, exp_no, params.params, deconv=False, **kwargs)
    results = fit.finalFit(vary_taus=vary,maxfev=maxfev)
    return results

def globalFitSumGausExponetial(x, data, *taus, vary=True, fwhm=0.12, tau_inf=1E12, t0=0, vary_t0= True, 
                               vary_fwhm=False, maxfev=5000, GVD_corrected=True, **kwargs):
    taus=list(taus)
    if type(vary) == bool:
        vary = [vary for i in taus]
    exp_no=len(taus)
    _ , n_traces = data.shape
    params=GlobExpParameters(n_traces, taus)
    params.adjustParams(t0,vary_t0,fwhm,vary_fwhm,GVD_corrected,tau_inf)
    fit=GlobalFitExponetial(x, data, exp_no, params.params, vary=True, deconv=True, 
                            tau_inf = tau_inf, GVD_corrected = GVD_corrected, **kwargs)
    results = fit.finalFit(vary_taus=vary,maxfev=maxfev)
    return results    

class GlobalFitExponetial(lmfit.Minimizer,ModelCreator):
    def __init__(self, x, data, exp_no, params,deconv=True,tau_inf=1E+12,GVD_corrected=True,**kwargs):
        weights=dict({'apply':False,'vector':None,'range':[],'type':'constant','value':2},**kwargs)
        self.weights = weights
        self.x = x
        self.data = data
        self.params = params
        self.deconv = deconv
        self.tau_inf = tau_inf
        self.exp_no = exp_no
        self.GVD_corrected = GVD_corrected
        self._number_it = 0
        self._prefit_done = False
        ModelCreator.__init__(self,self.exp_no,self.x,self.tau_inf)
        lmfit.Minimizer.__init__(self,self._objectiveExponential, params, nan_policy='propagate') 
    
    
    def preFit(self):
        fit_params = self.params.copy()
        ndata, nx = self.data.shape
        for iy in range(nx,0,-1):#range is decending just for chcking if it will work 
            single_param=lmfit.Parameters()
            single_param['y0_%i' %iy]=fit_params['y0_%i' %iy]
            single_param.add(('t0_%i' %iy), value=fit_params['t0_1'].value,expr=None,vary=fit_params['t0_1'].vary)
            if self.deconv:
                single_param['fwhm_%i' %iy]=fit_params['fwhm_1']
                if self.tau_inf is not None:
                    single_param['yinf_%i' %iy]=fit_params['yinf_%i' %iy]
            for i in range(self.exp_no):
                single_param.add(('tau%i_' %(i+1) +str (iy)), value=fit_params['tau%i_1' %(i+1)].value,expr=None,vary=False)
                single_param.add(('pre_exp%i_' %(i+1) +str (iy)),value=fit_params['pre_exp%i_' %(i+1) +str (iy)].value,vary=True)
            if self.deconv:
                result=lmfit.minimize(self._singleFit,single_param,args=(self.expNGaussDataset, iy-1),nan_policy='propagate')
            else:
                result=lmfit.minimize(self._singleFit,single_param,args=(self.expNDataset, iy-1),nan_policy='propagate')
            fit_params['y0_%i' %iy]=result.params['y0_%i' %iy]   
            for i in range(self.exp_no):
                fit_params['pre_exp%i_' %(i+1) +str (iy)]=result.params['pre_exp%i_' %(i+1) +str (iy)]
            if self.deconv:
                if self.GVD_corrected==False:
                    fit_params['t0_%i' %iy]=result.params['t0_%i' %iy]
                if self.tau_inf is not None:
                    fit_params['yinf_%i' %iy]=result.params['yinf_%i' %iy]    
            self.params=fit_params
            self._prefit_done=True                  
    
    def finalFit(self,vary_taus=True,maxfev=None,time_constraint=False,apply_weights=False):
        if type(vary_taus) == bool:
            vary_taus=[vary_taus for i in range(self.exp_no)]
        self.Fit_completed = False
        if self._prefit_done == False:
            self.preFit()
        fit_condition = [maxfev,time_constraint,'Exponential']#self.type_fit is important to know if we are doing an expoential or taget fit
        fit_params=self.params
        if time_constraint:
            for i in range (self.exp_no):
                if i == 0:
                    fit_params['tau%i_1' %(i+1)].min=fit_params['fwhm_1'].value
                else:
                    fit_params['tau%i_1' %(i+1)].min=fit_params['tau%i_1' %(i)].value 
        if apply_weights and len(self.weights['vector'])==len(self.x):
            self.weights['apply']=True
            fit_condition.append(self.weights)
        else:
            fit_condition.append('no weights')
        if maxfev != None:
            resultados = self.minimize(params=fit_params,maxfev=maxfev)
        else:
            resultados = self.minimize(params=fit_params)
        resultados = self._addToResultados(resultados,fit_condition)
        self._number_it = 0
        self.Fit_completed=True    
        if type(fit_condition[3]) == dict:
            self.weights['apply']=False
        return resultados
    
    def _addToResultados(self,resultados, fit_condition):
        resultados.x=self.x
        resultados.data=self.data
        resultados.wavelength=np.array([i for i in range(1,self.data.shape[1]+1)])
        tau_inf=self.tau_inf if self.deconv else None
        resultados.details={'exp_no':self.exp_no,'deconv':self.deconv,'type':'Exponential','tau_inf':tau_inf,\
                           'maxfev':fit_condition[0],'time_constraint':fit_condition[1],'svd_fit':False,'derivate':False}
        resultados.weigths = False if self.weights['apply'] == True else self.weights
        return resultados
        
    def _singleFit(self,params,function,i):
        """does a fit of a single trace"""
        if self.deconv:
            return self.data[:, i] - function(params, i)
        else:
            t0 = params['t0_%i'%(i+1)].value
            index=np.argmin([abs(i-t0) for i in self.x])
            return self.data[index:, i] - function(params, i)

    def _objectiveExponential(self,params):
        if self.deconv:
            if self.GVD_corrected:
               t0 = params['t0_1'].value
               fwhm= params ['fwhm_1'].value
               values=[params['tau%i_1' %(ii+1)].value for ii in range(self.exp_no)]  
               if self.tau_inf is not None: values.append(self.tau_inf)
               expvects = [self.expGauss(self.x-t0,tau,fwhm/2.35482) for tau in values]
               resid=self._generateResidues(self.expNGaussDatasetFast,params,expvects) 
            else:
               resid=self._generateResidues(self.expNGaussDataset,params) 
        else:
            t0 = params['t0_1'].value
            index=np.argmin([abs(i-t0) for i in self.x])
            values=[params['tau%i_1' %(ii+1)].value for ii in range(self.exp_no)] 
            expvects=[self.exp1(self.x-t0,tau) for tau in values]
            resid=self._generateResidues(self.expNDatasetFast,params,expvects)[index:,:] 
            
        self._number_it=self._number_it+1
        if(self._number_it % 100 == 0):
            print(self._number_it)
            print(sum(np.abs(resid.flatten())))
        return resid.flatten()
    
    def _generateResidues(self,funtion,params,extra_param=None):
        ndata, nx = self.data.shape
        data=self.data[:]
        resid=data*1.0
        if extra_param is not None:
            for i in range(nx):
                resid[:, i] = data[:, i] - funtion(params, i, extra_param) 
        else:
            for i in range(nx):
                resid[:, i] = data[:, i] - funtion(params, i) 
        if self.weights['apply']==True: 
            resid[:, i]=resid[:, i]*self.weights['vector']
        return resid    
       