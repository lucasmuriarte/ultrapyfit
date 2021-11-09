import unittest
from ultrafast.old.ExponentialFit import globalfit_exponential, \
    globalfit_gauss_exponential
from ultrafast.utils.divers import get_root_directory, read_data, select_traces
import numpy as np
import os


class TestGlobalFitClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(get_root_directory(), 'examples/data/denoised_2.csv')
        path2 = os.path.join(get_root_directory(), 'examples/data/gauss_denoised.csv')

        cls.original_taus = [8, 30, 200]

        cls.time, cls.data, cls.wave = read_data(path, wave_is_row=True)
        cls.data_select, cls.wave_select = select_traces(cls.data, cls.wave, 'auto')

        cls.time_gauss, cls.data_gauss, cls.wave_gauss = read_data(path2, wave_is_row=True)
        cls.data_select_gauss, cls.wave_select_gauss = select_traces(cls.data_gauss, cls.wave_gauss, 10)

    def test_globalfit_exponential(self):
        result = globalfit_exponential(self.time, self.data_select, 4, 40, 400)
        params_result = result.params
        final_taus = [
            params_result['tau1_1'].value,
            params_result['tau2_1'].value,
            params_result['tau3_1'].value
        ]
        self.assertTrue(self.assertNearlyEqualArray(self.original_taus, final_taus, 7))

    def test_globalfit_gauss_exponential(self):
        result = globalfit_gauss_exponential(
            self.time_gauss, self.data_select_gauss, 8, 40, 400,
            fwhm=0.12, vary_fwhm=True, vary_t0=False, tau_inf=1E12)

        params_result = result.params
        final_taus = [params_result['tau1_1'].value, params_result['tau2_1'].value, params_result['tau3_1'].value]
        self.assertTrue(self.assertNearlyEqualArray(self.original_taus, final_taus, 3))

    @staticmethod
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


if __name__ == '__main__':
    unittest.main()
