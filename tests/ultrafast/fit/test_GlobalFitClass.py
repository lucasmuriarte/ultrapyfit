import unittest
from ultrafast.old.ExponentialFit import globalfit_exponential, \
    globalfit_gauss_exponential
from ultrafast.utils.divers import read_data, select_traces
import numpy as np


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

    @classmethod
    def setUpClass(self):
        path = 'examples/data/denoised_2.csv'
        path2 = 'examples/data/gauss_denoised.csv'

        self.original_taus = [8, 30, 200]

        self.time, self.data, self.wave = read_data(path, wave_is_row=True)
        self.data_select, self.wave_select = select_traces(self.data, self.wave, 'auto')

        self.time_gauss, self.data_gauss, self.wave_gauss = read_data(path2, wave_is_row=True)
        self.data_select_gauss, self.wave_select_gauss = select_traces(self.data_gauss, self.wave_gauss, 10)

    def test_globalfit_exponential(self):
        result = globalfit_exponential(self.time, self.data_select, 4, 40, 400)
        params_result = result.params
        final_taus = [
            params_result['tau1_1'].value,
            params_result['tau2_1'].value,
            params_result['tau3_1'].value
        ]
        self.assertTrue(assertNearlyEqualArray(self.original_taus, final_taus, 7))

    def test_globalfit_gauss_exponential(self):
        result = globalfit_gauss_exponential(
            self.time_gauss, self.data_select_gauss, 8, 40, 400,
            fwhm=0.12, vary_fwhm=True, vary_t0=False, tau_inf=1E12)

        params_result = result.params
        final_taus = [params_result['tau1_1'].value, params_result['tau2_1'].value, params_result['tau3_1'].value]
        self.assertTrue(assertNearlyEqualArray(self.original_taus, final_taus, 5))


if __name__ == '__main__':
    unittest.main()
