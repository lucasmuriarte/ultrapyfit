# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:26:45 2021

@author: lucas
"""
import unittest
import numpy as np
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.fit.GlobalParams import GlobExpParameters
from parameterized import parameterized
from math import e as euler
from scipy.special import erfc

taus = [8,30,200]
pre_exp = [0.1,0.01,0.001]
values = [[pre_exp[i],taus[i]] for i in range(len(taus))]
number_exp = 3
time = np.linspace(5,106,101)
test_model_creator = ModelCreator(number_exp,time)

def expModifiedGaussian(x, amplitude=1, center=0, sigma=1.0, tau=1.0):
    """
    an alternative exponentially modified Gaussian from:
    https://stackoverflow.com/questions/54197603/the-use-of-lmfit-exponentialgaussianmodel
    """
    dx = center-x
    return amplitude* np.exp(-dx/tau) * erfc( dx/(np.sqrt(2)*sigma))



class TestModelCreator(unittest.TestCase):
    """
    test for ModelCreator Class
    """
    
    def assertEqualArray(self, array1, array2):
        """
        returns "True" if all elements of two arrays are identical
        """
        value = (array1==array2).all()
        return value
    
    def assertNearlyEqualArray(self, array1, array2, decimal):
        """ 
        returns "True" if all elements of two arrays 
        are identical until decimal
        """
        dif = np.array(array1) - np.array(array2)
        value = (dif < 1**(-decimal)).all()
        return value
        
    @parameterized.expand([[i] for i in taus])
    def test_exp1(self, tau):
        result = test_model_creator.exp1(time, tau)
        exp = euler**(-time/tau)
        self.assertEqual(len(result),len(exp))
        self.assertTrue(self.assertNearlyEqualArray(result, exp, 8))
    
    @parameterized.expand([[0.12,0,values],
                           [0.18,5,values],
                           [15,-3,values]])
    def test_expN(self, y0, t0, values):
        result = test_model_creator.expN(time, y0, t0, values)
        exp = y0 + sum([pre_exp*euler**(-(time - t0)/tau)
                        for pre_exp,tau in values])
        self.assertEqual(len(result),len(exp))
        self.assertTrue(self.assertNearlyEqualArray(result, exp, 8))
    
    @parameterized.expand([[i,np.random.random_sample()] for i in taus])
    def test_expGauss(self, tau, sigma):
        result = test_model_creator.expGauss(time, tau, sigma)
        exp = expModifiedGaussian(time, tau=tau, sigma = sigma)
        self.assertEqual(len(result),len(exp))
        self.assertTrue(self.assertNearlyEqualArray(result, exp, 8))
    
    @parameterized.expand([[0.12, 0, np.random.random_sample(), values],
                           [0.18, 5, np.random.random_sample(), values],
                           [15, -3, np.random.random_sample(), values]])
    def test_expNGauss(self, y0, t0, fwhm, values):
        result = test_model_creator.expNGauss(time, y0, t0, fwhm, values)
        exp = y0+sum([expModifiedGaussian(time, amplitude = pre_exp, center = t0, 
                                          sigma = fwhm/2.35482, tau = tau) 
                      for pre_exp,tau in values])
        self.assertEqual(len(result), len(exp))
        self.assertTrue(self.assertNearlyEqualArray(result, exp, 8))
    
    @parameterized.expand([[1, 5],
                           [3, 10],
                           [4, 20]])
    def test_expNDataset(self, i, t0):
        params = GlobExpParameters(5, taus)
        #generate parameters
        params.adjustParams(t0, vary_t0 = True, fwhm = None) 
        parametros = params.params
        # modify parameters to verify function
        for ii in range(1,len(taus)+1):
            parametros['pre_exp%i_' % ii +str(i)].value = \
                        pre_exp[ii-1]*i
        result = test_model_creator.expNDataset(parametros, i)
        # modify parameters to verify function
        index=np.argmin([abs(i-t0) for i in time])
        values = [[pre_exp[ii]*i,taus[ii]] for ii in range(len(taus))]
        exp = test_model_creator.expN(time[index:],0,t0,values)
        self.assertEqual(len(result),len(time[index:]))
        self.assertTrue(self.assertNearlyEqualArray(result,exp,20))
        
    @parameterized.expand([[1, 5, 1E6],
                           [3, 10, None],
                           [4, 20, 1E12]])
    def test_expNGaussDataset(self, i, t0, tau_inf):
        params = GlobExpParameters(5,taus)
        #generate parameters
        params.adjustParams(t0, vary_t0 = True, fwhm = 0.12, tau_inf = tau_inf) 
        parametros = params.params
        test_model_creator.tau_inf = tau_inf
        # modify parameters to verify function
        for ii in range(1,len(taus)+1):
            parametros['pre_exp%i_' % ii +str(i)].value = \
                        pre_exp[ii-1]*i
        result = test_model_creator.expNGaussDataset(parametros,i)
        # modify parameters to verify function
        # index=np.argmin([abs(i-t0) for i in time])
        values = [[pre_exp[ii]*i,taus[ii]] for ii in range(len(taus))]
        if tau_inf != None:
            values.append([0.001, tau_inf])
        exp = test_model_creator.expNGauss(time,0,t0,0.12,values)
        self.assertEqual(len(result),len(time))
        self.assertTrue(self.assertNearlyEqualArray(result,exp,20))
    
    @parameterized.expand([[1, 5],
                           [3, 10],
                           [4, 20]])
    def test_expNDatasetFast(self, i, t0):
        params = GlobExpParameters(5, taus)
        #generate parameters
        params.adjustParams(t0, vary_t0 = True, fwhm = None) 
        parametros = params.params
        index=np.argmin([abs(i-t0) for i in time])
        expvects=[test_model_creator.exp1(time[index:]-t0,tau) for tau in taus]
        # modify parameters to verify function
        for ii in range(1,len(taus)+1):
            parametros['pre_exp%i_' % ii +str(i)].value = \
                        pre_exp[ii-1]*i
        result = test_model_creator.expNDatasetFast(parametros, i, expvects)
        values = [[pre_exp[ii]*i,taus[ii]] for ii in range(len(taus))]
        exp = test_model_creator.expN(time[index:],0,t0,values)
        self.assertEqual(len(result),len(time[index:]))
        self.assertTrue(self.assertNearlyEqualArray(result,exp,20))    
        
    @parameterized.expand([[1, 5, 1E6],
                            [3, 10, None],
                            [4, 20, 1E12]])
    def test_expNGaussDatasetFast(self, i, t0, tau_inf):
        params = GlobExpParameters(5, taus)
        #generate parameters
        params.adjustParams(t0, vary_t0=True, fwhm=0.12, tau_inf=tau_inf)
        parametros = params.params
        test_model_creator.tau_inf = tau_inf
        expvects=[test_model_creator.expGauss(
            time-t0,tau,parametros['fwhm_'+str(i)]/2.35482) for tau in taus]
        # modify parameters to verify function
        for ii in range(1,len(taus)+1):
            parametros['pre_exp%i_' % ii +str(i)].value = \
                        pre_exp[ii-1]*i
        values = [[pre_exp[ii]*i,taus[ii]] for ii in range(len(taus))]
        if tau_inf != None:
            values.append([0.001, tau_inf])
            expvects.append(test_model_creator.expGauss(
            time-t0,tau_inf,parametros['fwhm_'+str(i)]/2.35482))
        result = test_model_creator.expNGaussDatasetFast(parametros, i, expvects)
        exp = test_model_creator.expN(time,0,t0,values)
        self.assertEqual(len(result),len(time))
        self.assertTrue(self.assertNearlyEqualArray(result,exp,20))
        
if __name__ == '__main__':
    unittest.main()    
    
   # def assertNearlyEqualArray(array1,array2,decimal):
   #  value = array1-array2