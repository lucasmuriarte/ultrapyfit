import unittest
import pickle
from ultrafast.fit.GlobalFitBootstrap import BootStrap
from parameterized import parameterized
from ultrafast.fit.ExponentialFit import GlobalFitExponential
from ultrafast.utils.Preprocessing import ExperimentException
from copy import deepcopy

path = 'tests/ultrafast/fit/test_result.obj'

with open(path, 'rb') as file:
    result = pickle.load(file)

boot = BootStrap(result)
data = result.data


class TestBootStrap(unittest.TestCase):

    @parameterized.expand([[10], [7], [15]])
    def test__data_sets_from_data(self, n_boots):
        res = boot._data_sets_from_data(n_boots)
        self.assertEqual(data.shape, res[:, :, 1].shape)
        self.assertEqual(n_boots, res.shape[2])
    
    @parameterized.expand([[10], [7], [15]])
    def test__data_sets_from_residues(self, n_boots):
        res = boot._data_sets_from_residues(n_boots)
        self.assertEqual(data.shape, res[:, :, 1].shape)
        self.assertEqual(n_boots, res.shape[2])

    def test__get_original_fitter(self):
        params2 = deepcopy(result.estimation_params)
        for i in params2:
            params2[i].value = params2[i].init_value
        fitter, params = boot._get_original_fitter()
        self.assertEqual(fitter, GlobalFitExponential)
        self.assertEqual(params,  params2)

    def test__details(self):
        exp_no, type_fit, deconv, maxfev, tau_inf = boot._details()
        self.assertEqual(result.details['exp_no'], exp_no)
        self.assertEqual(result.details['type'], type_fit)
        self.assertEqual(result.details['deconv'], deconv)
        self.assertEqual(result.details['maxfev'], maxfev)
        self.assertEqual(result.details['tau_inf'], tau_inf)

    @parameterized.expand([[10], [15], [20], [25], [33], [50], [14], [38]])
    def test__get_division_number(self, val):
        if val in [10, 15, 20, 25, 33, 50]:
            div = boot._get_division_number(val)
            self.assertEqual(div, round(100/val))
        else:
            with self.assertRaises(ExperimentException) as context:
                boot._get_division_number(val)
            msg = 'Size should be either 10, 15, 20, 25, 33, this is ' \
                  'the values in percentage that will be randomly changed'
            self.assertTrue(msg in str(context.exception))

    def test__get_fit_params_names(self):
        exp_no, type_fit, deconv, maxfev, tau_inf = boot._details()
        names = boot._get_fit_params_names(type_fit, exp_no, deconv)
        self.assertEqual(names, ['t0_1', 'tau1_1', 'tau2_1', 'tau3_1'])
    
    def test__generate_pandas_results_dataframe(self):
        pd_frame = boot._generate_pandas_results_dataframe()
        pd_keys = [i for i in pd_frame.keys()]
        df = ['N° Iterations', 'red Xi^2', 'success']
        suposed_keys = ['tau1 initial', 'tau1 final', 'tau2 initial', 
                        'tau2 final', 'tau3 initial', 'tau3 final'] + df
        self.assertEqual(pd_keys, suposed_keys)
    
    @parameterized.expand([[['t0_1', 'tau1_1', 'tau2_1', 'tau3_1'], True],
                           [['t0_1', 'tau1_1', 'tau3_1'], False],
                           [['t0_1', 'tau3_1'], True]])
    def test__initial_final_values(self, names, only_vary):
        params = result.estimation_params
        initial_values = [params[name].init_value for name in names]
        final_values = [params[name].value for name in names]
        ini, final = boot._initial_final_values(result.estimation_params,
                                                names, only_vary)
        if only_vary:
            initial_values = initial_values[1:]
            final_values = final_values[1:]
        self.assertEqual(final, final_values)
        self.assertEqual(ini, initial_values)

    @parameterized.expand([[3, 25, 'residues'],
                           [8, 33, 'residues'],
                           [6, 10, 'data']])
    def test_generate_data_sets(self, n_boots, size, data_from):
        boot_2 = BootStrap(result)
        data2 = boot_2.generate_data_sets(n_boots, size, data_from, True)
        self.assertEqual(data_from, boot_2.data_simulated)
        self.assertEqual(data2.shape[2], n_boots)
        if size == 'residues':
            self.assertEqual(data2._size, size)

    def test_fit_bootstrap(self):
        boot_2 = BootStrap(result)
        boot_2.generate_data_sets(3, 25, 'data')
        boot_2.fit_bootstrap()
        for i in range(1, boot_2.bootstrap_result.shape[0]+1):
            val1 = round(boot_2.bootstrap_result['tau1 final'][i])
            val2 = round(boot_2.bootstrap_result['tau2 final'][i])
            val3 = round(boot_2.bootstrap_result['tau3 final'][i])
            self.assertEqual(boot_2.bootstrap_result['tau1 initial'][i], 4)
            self.assertEqual(val1, 8)
            self.assertEqual(boot_2.bootstrap_result['tau2 initial'][i], 60)
            self.assertEqual(val2, 30)
            self.assertEqual(boot_2.bootstrap_result['tau3 initial'][i], 500)
            self.assertEqual(val3, 200)
        self.assertEqual(data.shape[2], 3)
        
        
if __name__ == '__main__':
    unittest.main()
