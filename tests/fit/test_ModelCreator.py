import unittest
import numpy as np
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.fit.GlobalParams import GlobExpParameters
from parameterized import parameterized
from math import e as euler
from scipy.special import erfc
from ultrafast.utils.test_tools import ArrayTestCase


taus = [8, 30, 200]


def expModifiedGaussian(x, amplitude=1, center=0, sigma=1.0, tau=1.0):
    """
    an alternative exponentially modified Gaussian from:
    https://stackoverflow.com/questions/54197603/the-use-of-lmfit-exponentialgaussianmodel
    """
    dx = center - x
    return amplitude * np.exp(- dx / tau) * erfc(dx / (np.sqrt(2) * sigma))


class TestModelCreator(ArrayTestCase):
    """
    test for ModelCreator Class
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.number_exp = 3
        cls.time = np.linspace(5, 106, 101)
        cls.test_model_creator = ModelCreator(cls.number_exp, cls.time)
        
        cls.taus = taus
        cls.pre_exp = [0.1, 0.01, 0.001]
        cls.values = [[cls.pre_exp[i], cls.taus[i]] for i in range(len(taus))]
        
    @parameterized.expand([[i] for i in taus])
    def test_exp1(self, tau):
        result = self.test_model_creator.exp1(self.time, tau)
        exp = euler ** (- self.time / tau)

        self.assertEqual(len(result), len(exp))
        self.assertNearlyEqualArray(result, exp, 8)
    
    @parameterized.expand([
        [0.12, 0],
        [0.18, 5],
        [15, -3]])
    def test_expN(self, y0, t0):
        result = self.test_model_creator.expN(self.time, y0, t0, self.values)
        exp = y0 + sum(
            [pre_exp * euler ** (- (self.time - t0) / tau)
                for pre_exp, tau in self.values]
        )

        self.assertEqual(len(result), len(exp))
        self.assertNearlyEqualArray(result, exp, 8)
    
    @parameterized.expand([[i, np.random.random_sample()] for i in taus])
    def test_expGauss(self, tau, sigma):
        result = self.test_model_creator.expGauss(self.time, tau, sigma)
        exp = expModifiedGaussian(self.time, tau=tau, sigma=sigma)

        self.assertEqual(len(result), len(exp))
        self.assertNearlyEqualArray(result, exp, 8)
    
    @parameterized.expand([[0.12, 0, np.random.random_sample()],
                           [0.18, 5, np.random.random_sample()],
                           [15, -3, np.random.random_sample()]])
    def test_expNGauss(self, y0, t0, fwhm):
        result = self.test_model_creator.expNGauss(
            self.time, y0, t0, fwhm, self.values)

        exp = y0 + sum([
            expModifiedGaussian(
                self.time,
                amplitude=pre_exp,
                center=t0,
                sigma=fwhm / 2.35482,
                tau=tau
            ) for pre_exp, tau in self.values
        ])

        self.assertEqual(len(result), len(exp))
        self.assertNearlyEqualArray(result, exp, 8)
    
    @parameterized.expand([
        [1, 5],
        [3, 10],
        [4, 20]])
    def test_expNDataset(self, i, t0):
        params = GlobExpParameters(5, self.taus)
        
        # generate parameters
        params.adjustParams(t0, vary_t0=True, fwhm=None)
        params = params.params

        # modify parameters to verify function
        for ii in range(1, len(self.taus) + 1):
            params['pre_exp%i_' % ii + str(i)].value = self.pre_exp[ii - 1] * i

        result = self.test_model_creator.expNDataset(params, i)

        # modify parameters to verify function
        index = np.argmin([abs(i - t0) for i in self.time])
        values = [
            [self.pre_exp[ii] * i, self.taus[ii]]
            for ii in range(len(self.taus))
        ]

        exp = self.test_model_creator.expN(self.time[index:], 0, t0, values)

        self.assertEqual(len(result), len(self.time[index:]))
        self.assertNearlyEqualArray(result, exp, 10)
        
    @parameterized.expand([
        [1, 5, 1E6],
        [3, 10, None],
        [4, 20, 1E12]])
    def test_expNGaussDataset(self, i, t0, tau_inf):
        params = GlobExpParameters(5, self.taus)

        # generate parameters
        params.adjustParams(t0, vary_t0=True, fwhm=0.12, tau_inf=tau_inf)
        params = params.params

        self.test_model_creator.tau_inf = tau_inf

        # modify parameters to verify function
        for ii in range(1, len(self.taus) + 1):
            params['pre_exp%i_' % ii + str(i)].value = self.pre_exp[ii - 1] * i

        result = self.test_model_creator.expNGaussDataset(params, i)

        # modify parameters to verify function
        # index = np.argmin([abs(i - t0) for i in time])
        values = [
            [self.pre_exp[ii] * i, self.taus[ii]]
            for ii in range(len(self.taus))
        ]

        if tau_inf is not None:
            values.append([0.001, tau_inf])

        exp = self.test_model_creator.expNGauss(self.time, 0, t0, 0.12, values)

        self.assertEqual(len(result), len(self.time))
        self.assertNearlyEqualArray(result, exp, 10)
    
    @parameterized.expand([
        [1, 5],
        [3, 10],
        [4, 20]])
    def test_expNDatasetFast(self, i, t0):
        params = GlobExpParameters(5, self.taus)

        # generate parameters
        params.adjustParams(t0, vary_t0=True, fwhm=None)
        params = params.params

        index = np.argmin([abs(i - t0) for i in self.time])
        expvects = [
            self.test_model_creator.exp1(self.time[index:] - t0, tau)
            for tau in self.taus
        ]

        # modify parameters to verify function
        for ii in range(1, len(self.taus) + 1):
            params['pre_exp%i_' % ii + str(i)].value = \
                self.pre_exp[ii - 1] * i

        result = self.test_model_creator.expNDatasetFast(
            params, i, expvects)
            
        values = [
            [self.pre_exp[ii] * i, self.taus[ii]]
            for ii in range(len(self.taus))
        ]

        exp = self.test_model_creator.expN(self.time[index:], 0, t0, values)

        self.assertEqual(len(result), len(self.time[index:]))
        self.assertNearlyEqualArray(result, exp, 10)
        
    @parameterized.expand([
        [1, 5, 1E6],
        [3, 10, None],
        [4, 20, 1E12]])
    def test_expNGaussDatasetFast(self, i, t0, tau_inf):
        params = GlobExpParameters(5, self.taus)
        # generate parameters
        params.adjustParams(t0, vary_t0=True, fwhm=0.12, tau_inf=tau_inf)
        params = params.params

        self.test_model_creator.tau_inf = tau_inf

        expvects = [
            self.test_model_creator.expGauss(
                self.time - t0, tau, params['fwhm_' + str(i)] / 2.35482
            ) for tau in self.taus
        ]

        # modify parameters to verify function
        for ii in range(1, len(self.taus) + 1):
            params['pre_exp%i_' % ii + str(i)].value = \
                self.pre_exp[ii - 1] * i

        values = [
            [self.pre_exp[ii] * i, self.taus[ii]]
            for ii in range(len(self.taus))
        ]

        if tau_inf is not None:
            values.append([0.001, tau_inf])
            expvects.append(
                self.test_model_creator.expGauss(
                    self.time - t0, tau_inf, params['fwhm_' + str(i)] / 2.35482
                )
            )

        result = self.test_model_creator.expNGaussDatasetFast(
            params, i, expvects)

        exp = self.test_model_creator.expN(self.time, 0, t0, values)

        self.assertEqual(len(result), len(self.time))
        self.assertNearlyEqualArray(result, exp, 20)
        

if __name__ == '__main__':
    unittest.main()
