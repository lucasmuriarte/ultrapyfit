import unittest
from ultrafast.fit.GlobalFit import *
from parameterized import parameterized
from ultrafast.utils.divers import read_data, select_traces
import numpy as np
from ultrafast.fit.ModelCreator import ModelCreator
from ultrafast.fit.GlobalParams import GlobExpParameters
from copy import deepcopy


time_simulated = np.logspace(0, 3, 150)

path = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_denoised_2.csv'
original_taus = [8, 30, 200]

time, data, wave = read_data(path, wave_is_row=True)
data_select, wave_select = select_traces(data, wave, 'auto')


param_generator = GlobExpParameters(10, [8, 30, 200])
param_generator.adjustParams(0, fwhm=None)
params = param_generator.params
model = ModelCreator(3, time_simulated)
data_simulated = np.zeros((150, 10))
for i in range(10):
    data_simulated[:, i] = model.expNDataset(params, i)


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


class TestGlobalFit(unittest.TestCase):
    def test__generate_residues(self):
        fitter = GlobalFit(time_simulated, data_simulated, 3, params)
        res = fitter._generate_residues(fitter.expNDataset, params)
        self.assertTrue((res == 0).all())
        
    @parameterized.expand([[2], [7], [9]])
    def test__single_fit(self, val):
       fitter = GlobalFit(time_simulated, data_simulated, 3, params)
       res = fitter._single_fit(params, fitter.expNDataset, val) 
       self.assertTrue((res == 0).all())
       
    def test__get_fit_details(self):
        fitter = GlobalFit(time_simulated, data_simulated, 3, params)
        details = fitter._get_fit_details()
        self.assertTrue(type(details) == dict)


class TestContainer(unittest.TestCase):
    def test___init__(self):
        container = Container()
        container.data = 1
        self.assertTrue(hasattr(container, 'data'))


class TestGlobalFitResult(unittest.TestCase):
    def test_add_data_details(self):
        fitter = GlobalFit(time_simulated, data_simulated, 3, params)
        results = Container(params=params)
        results_test = GlobalFitResult(results)
        details = fitter._get_fit_details()
        results_test.add_data_details(fitter._data_ensemble, details)
        self.assertTrue(hasattr(results_test, 'data'))
        self.assertTrue(hasattr(results_test, 'x'))
        self.assertTrue(hasattr(results_test, 'wavelength'))
        self.assertTrue(hasattr(results_test, 'details'))
        self.assertTrue(hasattr(results_test, 'params'))

class TestGlobalFitExponential(unittest.TestCase):
    def test_global_fit(self):
        params = GlobExpParameters(data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params

        fitter = GlobalFitExponential(time, data_select, 3, parameters, False,
                                      wavelength=wave_select)
        result = fitter.global_fit()
        params_result = result.params
        final_taus = [params_result['tau1_1'].value,
                      params_result['tau2_1'].value,
                      params_result['tau3_1'].value]
        self.assertTrue(assertNearlyEqualArray(original_taus, final_taus, 7))
        
    def test__apply_time_constraint(self):
        params = GlobExpParameters(data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params

        fitter = GlobalFitExponential(time, data_select, 3, parameters, False,
                                      wavelength=wave_select)
        fitter._apply_time_constraint()
        params = fitter.params
        self.assertTrue(params['tau2_1'].min == params['tau1_1'].value)
        self.assertTrue(params['tau3_1'].min == params['tau2_1'].value)
        
    def test__uncontraint_times(self):
        params = GlobExpParameters(data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params

        fitter = GlobalFitExponential(time, data_select, 3, parameters, False,
                                      wavelength=wave_select)
        fitter._apply_time_constraint()
        fitter._uncontraint_times()
        params = fitter.params
        self.assertTrue(params['tau2_1'].min is None)
        self.assertTrue(params['tau3_1'].min is None)
        
    def test__objective(self):
        fitter = GlobalFitExponential(time_simulated, data_simulated, 3,
                                      params, False)
        res = fitter._objective(fitter.params)
        self.assertTrue((res == 0).all())

    def test__pre_fit(self):
        params = GlobExpParameters(data_select.shape[1], [4, 40, 400])
        params.adjustParams(0, False, None)
        parameters = params.params
        fitter = GlobalFitExponential(time, data_select, 3, parameters, False,
                                      wavelength=wave_select)
        fitter.pre_fit()
        param = fitter.params
        for par in param:
            if 'pre' in par:
                self.assertTrue(param[par].init_value != param[par].value)
            elif 'tau' in par:
                self.assertTrue(param[par].init_value == param[par].value)


if __name__ == '__main__':
    unittest.main()
