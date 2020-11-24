# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:35:15 2020

@author: lucas
"""
import numpy as np
from scipy.special import erf

class ModelCreator:
    def __init__(self,exp_no,time,tau_inf=1E+12,):
        self.exp_no=exp_no
        self.x=time
        self.tau_inf=tau_inf
        
    def exp1(self,x, tau):
        "basic gaussian"
        return np.exp(-x/tau) 
    
    def expN (self,time,y0,t0,values):
        """values should be a list of list containing the pre_exps and taus values"""
        return y0+sum([pre_exp*self.exp1(time-t0,tau) for pre_exp,tau in values])
    
    def expGauss(self,time,tau,sigma):
        inv_tau=1/tau
        return 0.5*np.exp(-inv_tau*time + sigma**2*inv_tau**2/2 )*(1+erf((time-sigma**2*inv_tau)/(sigma*2**0.5)))
    
    def expNGauss (self,time,y0,t0,fwhm,values):
        """values should be a list of list containing the pre_exps and taus values"""
        return y0+sum([(pre_exp)*self.expGauss(time-t0,tau,fwhm/2.35482) for pre_exp,tau in values])
    
    def expNGaussDataset(self,params, i):
        """calc 2 exponetial function from params for data set i
        using simple, hardwired naming convention"""
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        fwhm= params ['fwhm_%i' % (i+1)].value
        values=[[params['pre_exp%i_' % (ii+1)+str(i+1)].value,params['tau%i_' %(ii+1)+str(i+1)].value] for ii in range(self.exp_no)]            
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (i+1)].value
            values.append([yinf,self.tau_inf])
        return self.expNGauss(self.x,y0,t0,fwhm,values)
                 
    def expNDataset(self,params,i):
        """calc 2 exponetial function from params for data set i
        using simple, hardwired naming convention"""   
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        index=np.argmin([abs(i-t0) for i in self.x])
        values=[[params['pre_exp%i_' % (ii+1)+str(i+1)].value,params['tau%i_' %(ii+1)+str(i+1)].value] for ii in range(self.exp_no)]            
        return self.expN(self.x[index:],y0,t0,values)
    
    def expNDatasetFast(self,params,i,expvects):
        y0 = params['y0_%i' % (i+1)].value
        pre_exp=[params['pre_exp%i_' % (ii+1)+str(i+1)].value for ii in range(self.exp_no)] 
        return y0+sum([pre_exp[iii]*expvects[iii] for iii in range(self.exp_no)]) 
            
    def expNGaussDatasetFast(self, params, i, expvects):
        y0 = params['y0_%i' % (i+1)].value
        pre_exp=[params['pre_exp%i_' % (ii+1)+str(i+1)].value for ii in range(self.exp_no)] 
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (i+1)].value
            return y0+sum([pre_exp[iii]*expvects[iii] for iii in range(self.exp_no)])+yinf*expvects[-1]
        else:
            return y0+sum([pre_exp[iii]*expvects[iii] for iii in range(self.exp_no)])    
        
    def expNGaussDatasetTM(self,params, i,cons_eigen):
        exp_no=self.exp_no
        x=self.x
        deconv = self.deconv
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        pre_exp = [params ['pre_exp%i_' % (ii+1)+str (i+1)].value for ii in range(exp_no)]
        coeffs,eigs,eigenmatrix = cons_eigen[0],cons_eigen[1],cons_eigen[2]
        if deconv:
            fwhm = params ['fwhm_%i' % (i+1)].value
            expvects = [coeffs[val]*self.expGauss(x-t0,-1/eigs[val],fwhm/2.35482) for val in range(len(eigs))]
        else:
            expvects = [coeffs[val]*self.exp1(x-t0,-1/eigs[val]) for val in range(len(eigs))]
        concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
        return y0+sum([pre_exp[iii]*concentrations[iii] for iii in range(exp_no)])                
        