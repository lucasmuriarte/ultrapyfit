# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:35:15 2020

@author: lucas
"""
import numpy as np
from scipy.special import erf


class ModelCreator:
    """
    Class containing the simple exponential and sum of exponential functions
    (including modified Gaussian exponential). This functions are used to
    generate the model that is later fit to the data and the parameters
    optimized. The class has several static methods that returns the
    exponential functions from the parameters pass as well as methods that
    return exponential from the parameters inside an lmfit parameters object.
    This parameter lmfit object can be obtained from the GlobExpParameters
    class.

    static Methods
    --------------
    exp1: single exponential decay function

    expN: weighted sum of exponential decay functions with an off-set

    ExpGauss: exponential modified Gaussian function

    ExpNGauss: weighted sum of exponential modified Gaussian decay functions
    with an off-set
    """
    def __init__(self, exp_no, time, tau_inf=1E+12):
        """
        Constructor

        Parameters
        ----------
        exp_no: int
            number of exponential that the methods return (non-static)

        time: 1darray
            time vector

        tau_inf: float or int (default 1E12)
            extra time constant to evaluate Gauss modified exponential functions
            that have not completely decay at long decay times (off-set).
        """
        self.exp_no = exp_no
        self.x = time
        self.tau_inf = tau_inf

    @staticmethod
    def erfDerrivative(z):
        """
        calculate derrivative of the error function
        
        
        Parameters
        ----------
        z: float or array of floats
            function argument
       
        Returns
        ----------
        float or array of floats
        """
        
        return 2*np.exp(-np.square(z))/1.7724538509055159
    

    @staticmethod
    def exp1(time, tau):
        """
        single exponential decay function
        
        Parameters
        ----------
        time: array type
            the time vector
            
        tau: float or int
            decay associated time value
        
        Returns
        ----------
        1darray of size equal to time-vector  
        """
        return np.exp(-time/tau)

    @staticmethod
    def expN(time, y0, t0, values):
        """
        weighted sum of exponential decay functions with an off-set
        
        
        Parameters
        ----------
        time: array type
            the time vector
            
        y0: float or int
            off-set value at the longest decay time
        
        t0: float or int
            initial time from where the exponential value will be evaluated
        
        values: list of list
            values should be a list of list containing the pre_exps and
            decay associated time values (taus)
            [[0.1,8],[0.001,30]]
       
        Returns
        ----------
        1darray of size equal to time-vector 
        """
        return y0+sum([pre_exp*ModelCreator.exp1(time-t0, tau)
                       for pre_exp, tau in values])

    @staticmethod
    def expGauss(time, tau, sigma):
        """
        exponential modified Gaussian function

        Parameters
        ----------
        time: array type
            the time vector
            
        tau: float or int
            decay associated time value
        
        sigma: float or int
            variance of the gaussian distribution
       
        Returns
        ----------
        1darray of size equal to time-vector 
        """
        inv_tau = 1/tau
        erf_res = (1+erf((time-sigma**2*inv_tau)/(sigma*2**0.5)))
        return 0.5*np.exp(-inv_tau*time + sigma**2*inv_tau**2/2)*erf_res

    @staticmethod
    def expNGauss(time, y0, t0, fwhm, values):
        """
        weighted sum of exponential modified Gaussian decay functions
        with an off-set
        
        Parameters
        ----------
        time: array type
            the time vector
            
        y0: float or int
            off-set value at the shortest decay time
            (the off-set at long decay times can be evaluated
             adding to values a long decay associated time )
        
        t0: float or int
            initial time from where the exponential value will be evaluated
            
        fwhm: float or int
            full width half maximum of the gaussian distribution
            
        values: list of list
            values should be a list of list containing the pre_exps and 
            decay associated time values (taus)
            [[0.1,8],[0.001,30]]
       
        Returns
        ----------
        1darray of size equal to x attribute vector (time vector)
        """
        return y0+sum([pre_exp*ModelCreator.expGauss(time-t0, tau, fwhm/2.35482)
                       for pre_exp, tau in values])

    @staticmethod
    def expGaussDerrivativeSigma(time, tau, sigma):
        """
        exponential derrivative by sigma

        Parameters
        ----------
        time: array type
            the time vector
            
        tau: float or int
            decay associated time value
        
        sigma: float or int
            variance of the gaussian distribution
       
        Returns
        ----------
        1darray of size equal to time-vector 
        """
        
        inv_tau = 1/tau
        inv2_tau = inv_tau**2
        erf_part = 1+erf((time-sigma**2*inv_tau)/(sigma*2**0.5))
        exp_part = np.exp(-inv_tau*time + sigma**2*inv2_tau/2)
        
        return exp_part*erf_part*(sigma*inv2_tau)+exp_part*\
              ModelCreator.erfDerrivative((time-sigma**2*inv_tau)/(sigma*2**0.5))*\
              (-time/(sigma**2*2**0.5)-inv_tau/(2**0.5))
    
    def expNGaussDatasetJacobianByPreExp(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by pre_exp in a equivalent 
        way to the expNGaussDataset method.
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)
            
        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)
        
        Returns
        ----------
        1darray of size equal to time-vector 
        """    
        #i = lambda_i-1
        t0 = params['t0_%i' % (lambda_i)].value
        fwhm = params['fwhm_%i' % (lambda_i)].value
    
        tau = params['tau%i_' % (tau_j)+str(lambda_i)].value
    
        time = self.x-t0
        sigma = fwhm/2.35482
        
        inv_tau = 1/tau
        inv2_tau = inv_tau**2
        erf_part = 1+erf((time-sigma**2*inv_tau)/(sigma*2**0.5))
        exp_part = np.exp(-inv_tau*time + sigma**2*inv2_tau/2)

        return 0.5*exp_part*erf_part
    
    def expNGaussDatasetJacobianBySigma(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by tau in a equivalent 
        way to the expNGaussDataset method.
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)
            
        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)
        
        Returns
        ----------
        1darray of size equal to time-vector 
        """    
        #i = lambda_i-1
        #tau_j is intentionally ignored
        t0 = params['t0_%i' % (lambda_i)].value
        fwhm = params['fwhm_%i' % (lambda_i)].value

        values = [[params['pre_exp%i_' % (ii+1)+str(lambda_i)].value,
                   params['tau%i_' % (ii+1)+str(lambda_i)].value]
                  for ii in range(self.exp_no)]
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (lambda_i)].value
            values.append([yinf, self.tau_inf])
         
        return sum([0.5*pre_exp*ModelCreator.expGaussDerrivativeSigma(self.x-t0, 
                                                                     tau, 
                                                                     fwhm/2.35482)
                       for pre_exp, tau in values])    
    
    def expNGaussDatasetJacobianByTau(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by tau in a equivalent 
        way to the expNGaussDataset method.
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)
            
        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)
        
        Returns
        ----------
        1darray of size equal to time-vector 
        """    
        #i = lambda_i-1
        t0 = params['t0_%i' % (lambda_i)].value
        fwhm = params['fwhm_%i' % (lambda_i)].value
    
        pre_exp = params['pre_exp%i_' % (tau_j)+str(lambda_i)].value
        tau = params['tau%i_' % (tau_j)+str(lambda_i)].value
    
        time = self.x-t0
        sigma = fwhm/2.35482
        
        inv_tau = 1/tau
        inv2_tau = inv_tau**2
        inv3_tau = inv_tau**3
        erf_part = 1+erf((time-sigma**2*inv_tau)/(sigma*2**0.5))
        exp_part = np.exp(-inv_tau*time + sigma**2*inv2_tau/2)

        tmp = exp_part*erf_part*(inv2_tau*time - sigma**2*inv3_tau)+\
              exp_part*ModelCreator.erfDerrivative((time-sigma**2*inv_tau)/(sigma*2**0.5))*\
              (-sigma**2*inv2_tau)/(sigma*2**0.5)
        
        return 0.5*pre_exp*tmp
             
    def expNGaussDataset(self, params, i):
        """
        calculate a weighted sum of exponential modified Gaussian decay
        functions with an off-set from params for data set i using simple
        hardwired naming convention. This function can be used for datasets
        having different t0
        
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        i: int
            number corresponding to the specific trace
       
        Returns
        ----------
        1darray of size equal to time-vector 
        """
        
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        fwhm = params['fwhm_%i' % (i+1)].value
        values = [[params['pre_exp%i_' % (ii+1)+str(i+1)].value,
                   params['tau%i_' % (ii+1)+str(i+1)].value]
                  for ii in range(self.exp_no)]
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (i+1)].value
            values.append([yinf, self.tau_inf])
        return self.expNGauss(self.x, y0, t0, fwhm, values)
                 
    def expNDataset(self, params, i):
        """
        calculate a weighted sum of exponential decay functions with an
        off-set from params for data set i using simple hardwired naming 
        convention. This function can be used for datasets having different t0.
        
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        i: int
            number corresponding to the specific trace
       
        Returns
        ----------
        1darray of size equal to time-vector 
        """
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        index = np.argmin([abs(i-t0) for i in self.x])
        values = [[params['pre_exp%i_' % (ii+1)+str(i+1)].value,
                   params['tau%i_' % (ii+1)+str(i+1)].value]
                  for ii in range(self.exp_no)]
        return self.expN(self.x[index:], y0, t0, values)

    def expNDatasetIRF(self, params, i, IRF):
        """
        calculate a weighted sum of exponential decay functions with an
        off-set from params for data set i using simple hardwired naming
        convention. This function can be used for datasets having different t0.


        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        i: int
            number corresponding to the specific trace

        Returns
        ----------
        1darray of size equal to time-vector
        """
        y0 = params['y0_%i' % (i + 1)].value
        t0 = int(params['t0_%i' % (i + 1)].value)
        # print(t0)
        values = [[params['pre_exp%i_' % (ii + 1) + str(i + 1)].value,
                   params['tau%i_' % (ii + 1) + str(i + 1)].value]
                  for ii in range(self.exp_no)]
        sum_exp = sum([pre_exp * ModelCreator.exp1(self.x, tau)
                       for pre_exp, tau in values])
        # print(len(sum_exp))
        # print(len(IRF))
        # print(len(result))

        # padding for convolution
        sum_exp_pad = np.pad(sum_exp, len(sum_exp)//2, mode='minimum')
        IRF_pad = np.pad(IRF, len(IRF) // 2, mode='minimum')
        # convolution
        # x = np.fft.fft(sum_exp)
        # h = np.fft.fft(IRF)
        # result = np.real(np.fft.ifft(x * h))
        result = np.convolve(sum_exp_pad, IRF_pad, mode='same')
        # removing padding
        mini = len(sum_exp)//2
        maxi = int(len(sum_exp)*1.5)
        result = result[mini:maxi]

        # adjusting shift between IRF and data
        result_y = np.ones(len(result))
        result_y[t0:] = result[:len(result)-t0]
        return result_y + y0
    
    def expNDatasetFast(self, params, i, expvects):
        """
        calculate a weighted sum of exponential modified Gaussian decay
        functions with an off-set from params for data set i using simple
        hardwired naming convention and GlobExpParameters object. This 
        function can only be used for datasets having identical t0.
        (is computationally faster than expNDataset)
        
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        i: int
            number corresponding to the specific trace
        
        expvects: 1darray 
            sum of expN functions where time-vector should be: time-t0
        Returns
        ----------
        1darray of size equal to expvects
        """
        y0 = params['y0_%i' % (i+1)].value
        pre_exp = [params['pre_exp%i_' % (ii+1)+str(i+1)].value
                   for ii in range(self.exp_no)]
        return y0+sum([pre_exp[iii]*expvects[iii]
                       for iii in range(self.exp_no)])
            
    def expNGaussDatasetFast(self, params, i, expvects):
        """
        calculate a weighted sum of exponential decay functions with an
        off-set from params for data set i using simple hardwired naming 
        convention and GlobExpParameters object. This function can only be
        used for datasets having identical t0.
        (is computationally faster than expNGaussDataset)
        
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        i: int
            number corresponding to the specific trace
        
        expvects: 1darray 
            sum of expN functions where time-vector should be: time-t0
        
        Returns
        ----------
        1darray of size equal to expvects
        """
        y0 = params['y0_%i' % (i+1)].value
        pre_exp = [params['pre_exp%i_' % (ii+1)+str(i+1)].value
                   for ii in range(self.exp_no)]
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (i+1)].value
            return y0+sum([pre_exp[iii]*expvects[iii]
                           for iii in range(self.exp_no)])+yinf*expvects[-1]
        else:
            return y0+sum([pre_exp[iii]*expvects[iii]
                           for iii in range(self.exp_no)])
        
    def expNGaussDatasetTM(self, params, i, cons_eigen):
        """
        calculate kinetic reaction model from params for data set i using 
        simple hardwired naming convention and GlobTargetParameters object. 
        (Automatically detects id all traces have identical t0)
        
        
        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global 
          fitting several decay traces
        
        i: int
            number corresponding to the specific trace
        
        cons_eigen: list
            list containing eingen vectors, eigen values and eigenmatrix

        Returns
        ----------
        1darray of size equal to expvects
        """
        exp_no = self.exp_no
        x = self.x
        if 'fwhm_1' in params.keys():
            deconv = True
        else:
            deconv = False
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        pre_exp = [params['pre_exp%i_' % (ii+1)+str(i+1)].value
                   for ii in range(exp_no)]
        coeffs, eigs, eigenmatrix = cons_eigen[0], cons_eigen[1], cons_eigen[2]
        if deconv:
            fwhm = params['fwhm_%i' % (i+1)].value/2.35482
            expvects = [coeffs[val]*self.expGauss(x-t0, -1/eigs[val], fwhm)
                        for val in range(len(eigs))]
        else:
            expvects = [coeffs[val]*self.exp1(x-t0, -1/eigs[val])
                        for val in range(len(eigs))]
        concentrations = [sum([eigenmatrix[i, j]*expvects[j]
                               for j in range(len(eigs))])
                          for i in range(len(eigs))]
        return y0+sum([pre_exp[iii]*concentrations[iii]
                       for iii in range(exp_no)])

    
    
    