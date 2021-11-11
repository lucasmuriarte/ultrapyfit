import unittest
from ultrafast.fit.GlobalFit import *
from parameterized import parameterized
from ultrafast.utils.divers import read_data, select_traces
import numpy as np
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.fit.GlobalParams import GlobExpParameters
from ultrafast.utils.divers import get_root_directory
import os
from unittest.util import safe_repr

class TestGlobalFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.time_simulated = np.logspace(0, 3, 150)
        path = os.path.join(get_root_directory(), 'examples/data/denoised_2.csv')
        cls.original_taus = [8, 30, 200]

        cls.time, data, wave = read_data(path, wave_is_row=True)
        cls.data_select, cls.wave_select = select_traces(data, wave, 10)

        param_generator = GlobExpParameters(10, [8, 30, 200])
        param_generator.adjustParams(0, fwhm=None)
        cls.params = param_generator.params
        model = ModelCreator(3, cls.time_simulated)
        cls.data_simulated = np.zeros((150, 10))

        for i in range(10):
            cls.data_simulated[:, i] = model.expNDataset(cls.params, i)
        return super().setUpClass()

    def test__generate_residues(self):
        fitter = GlobalFit(self.time_simulated, self.data_simulated, 3, self.params)
        res = fitter._generate_residues(fitter.expNDataset, self.params)
        self.assertTrue((res == 0).all())
        
    @parameterized.expand([[2], [7], [9]])
    def test__single_fit(self, val):
       fitter = GlobalFit(self.time_simulated, self.data_simulated, 3, self.params)
       res = fitter._single_fit(self.params, fitter.expNDataset, val) 
       self.assertTrue((res == 0).all())
       
    def test__get_fit_details(self):
        fitter = GlobalFit(self.time_simulated, self.data_simulated, 3, self.params)
        details = fitter._get_fit_details()
        self.assertTrue(type(details) == dict)

    def assertNearlyEqualArray(self, array1, array2, decimal, msg=None):
        """
        returns "True" if all elements of two arrays
        are identical until decimal
        """
        if type(array1) == list:
            array1 = np.array(array1)
        if type(array2) == list:
            array2 = np.array(array2)
        dif = np.array(array1) - np.array(array2)
        expr = (dif < 10**(-decimal)).all()

        if not expr:
            msg = self._formatMessage(msg, 'arrays are not neary equal, '
                ' try changing the number of decimals, '
                f'here it is set to {decimal}')

            raise self.failureException(msg)


class TestContainer(unittest.TestCase):
    def test___init__(self):
        container = Container()
        container.data = 1
        self.assertTrue(hasattr(container, 'data'))

class TestGlobalFitResult(TestGlobalFit):
    def test_add_data_details(self):
        fitter = GlobalFit(self.time_simulated, self.data_simulated, 3, self.params)
        results = Container(params=self.params)
        results_test = GlobalFitResult(results)
        details = fitter._get_fit_details()
        results_test.add_data_details(fitter._data_ensemble, details)
        self.assertTrue(hasattr(results_test, 'data'))
        self.assertTrue(hasattr(results_test, 'x'))
        self.assertTrue(hasattr(results_test, 'wavelength'))
        self.assertTrue(hasattr(results_test, 'details'))
        self.assertTrue(hasattr(results_test, 'params'))

class TestGlobalFitExponential(TestGlobalFit):
    def test_global_fit(self):
        params = GlobExpParameters(self.data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params

        fitter = GlobalFitExponential(self.time, self.data_select, 3, parameters, False,
                                      wavelength=self.wave_select)
                                      
        result = fitter.global_fit()
        params_result = result.params
        final_taus = [params_result['tau1_1'].value,
                      params_result['tau2_1'].value,
                      params_result['tau3_1'].value]

        self.assertNearlyEqualArray(self.original_taus, final_taus, 7)
        
    def test__apply_time_constraint(self):
        params = GlobExpParameters(self.data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params

        fitter = GlobalFitExponential(self.time, self.data_select, 3, parameters, False,
                                      wavelength=self.wave_select)
        fitter._apply_time_constraint()
        params = fitter.params
        self.assertTrue(params['tau2_1'].min == params['tau1_1'].value)
        self.assertTrue(params['tau3_1'].min == params['tau2_1'].value)
        
    def test__uncontraint_times(self):
        params = GlobExpParameters(self.data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params

        fitter = GlobalFitExponential(self.time, self.data_select, 3, parameters, False,
                                      wavelength=self.wave_select)
        fitter._apply_time_constraint()
        fitter._unconstraint_times()
        params = fitter.params
        self.assertTrue(params['tau2_1'].min is None)
        self.assertTrue(params['tau3_1'].min is None)
        
    def test__objective(self):
        fitter = GlobalFitExponential(self.time_simulated, self.data_simulated, 3,
                                      self.params, False)
        res = fitter._objective(fitter.params)
        self.assertTrue((res == 0).all())

    def test__pre_fit(self):
        params = GlobExpParameters(self.data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params
        fitter = GlobalFitExponential(self.time, self.data_select, 3, parameters, False,
                                      wavelength=self.wave_select)
        fitter.pre_fit()
        param = fitter.params
        for par in param:
            if 'pre' in par:
                self.assertTrue(param[par].init_value != param[par].value)
            elif 'tau' in par:
                self.assertTrue(param[par].init_value == param[par].value)


if __name__ == '__main__':
    unittest.main()
