import unittest
from chempyspec.ultrafast.fit.ExponentialFit import globalfit_exponential, globalfit_gauss_exponential
from chempyspec.ultrafast.utils.utils import read_data, select_traces
import numpy as np


path = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_denoised_2.csv'
path2 = 'C:/Users/lucas/git project/ultrafast/examples/3_exp_data_gauss_denoised.csv'

original_taus = [8, 30, 200]

time, data, wave = read_data(path, wave_is_row=True)
data_select, wave_select = select_traces(data, wave, 'auto')

time_gauss, data_gauss, wave_gauss = read_data(path2, wave_is_row=True)
data_select_gauss, wave_select_gauss = select_traces(data_gauss, wave_gauss, 10)


def assertNearlyEqualArray(array1, array2, decimal):
    """
    returns "True" if all elements of two arrays
    are identical until decimal
    """
    if type(array1) == list:
        array1 = np.array(array1)
    if type(array2) == list:
        array2 = np.array(array2)
    dif = np.array(array1) - np.array(array2)
    value = (dif < 10**(-decimal)).all()
    return value


class TestGlobalFitClass(unittest.TestCase):

    def test_globalfit_exponential(self):
        result = globalfit_exponential(time, data_select, 4, 40, 400)
        params_result = result.params
        final_taus = [params_result['tau1_1'].value, params_result['tau2_1'].value, params_result['tau3_1'].value]
        self.assertTrue(assertNearlyEqualArray(original_taus, final_taus, 7))

    def test_globalfit_gauss_exponential(self):
        result = globalfit_gauss_exponential(time_gauss, data_select_gauss, 8, 40, 400, fwhm=0.12, vary_fwhm=True,
                                             vary_t0=False, tau_inf=1E12)
        params_result = result.params
        final_taus = [params_result['tau1_1'].value, params_result['tau2_1'].value, params_result['tau3_1'].value]
        self.assertTrue(assertNearlyEqualArray(original_taus, final_taus, 5))


if __name__ == '__main__':
    unittest.main()
