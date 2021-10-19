import unittest
from ultrafast.experiment import Experiment
import os
import numpy as np
from copy import deepcopy
from parameterized import parameterized


class TestExperiment(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExperiment, self).__init__(*args, **kwargs)
        path = '../../examples/data/exp3_data_denoised.csv'
        self.path_save = os.path.abspath("my_test")
        self.path = os.path.abspath(path)
        self.experiment = Experiment.load_data(self.path)
        self.original_time = deepcopy(self.experiment.time)

    def check_preprocessing_function(self, name, extra=None):
        # print(name)
        action_number = len(self.experiment.action_records.__dict__) - 3
        action = getattr(self.experiment.action_records, f"_{action_number}")
        # print(action)
        suppose_action = " ".join(name.split("_"))
        if extra is not None:
            suppose_action += f" {extra}"
        # print(action, suppose_action)
        action_written = action == suppose_action
        data_added = hasattr(self.experiment.data_sets, "before_"+name)
        report_updated = hasattr(self.experiment.preprocessing_report, name)
        # print(action_written, data_added, report_updated)
        values_true = (action_written, data_added,
                       report_updated) == (True, True, True)
        return values_true

    def test__initialized(self):
        self.assertEqual(hasattr(self.experiment.data_sets, "original_data"),
                         True)
        self.assertEqual(hasattr(self.experiment.fit_records, "single_fits"),
                         True)
        self.assertEqual(hasattr(self.experiment.fit_records, "bootstrap_record"),
                         True)
        self.assertEqual(hasattr(self.experiment.fit_records, "conf_interval"),
                         True)
        self.assertEqual(hasattr(self.experiment.fit_records, "target_models"),
                         True)
        self.assertEqual(hasattr(self.experiment.fit_records, "global_fits"),
                         True)
        self.assertEqual(hasattr(
            self.experiment.fit_records, "integral_band_fits"), True)
        self.assertEqual(hasattr(self.experiment, "_unit_formater"), True)

    def test_time(self):
        val = (self.experiment.time == self.experiment.x).all()
        self.assertEqual(val, True)

    def test_time_setter(self):
        new_time = np.random.randn(1000)
        self.experiment.time = new_time
        val = (self.experiment.x == new_time).all()
        self.assertEqual(val, True)
        self.experiment.x = self.original_time

    def test_save(self):
        self.experiment.save(self.path_save)
        is_file = os.path.isfile(self.path_save + ".exp")
        self.assertEqual(is_file, True)
        os.remove(os.path.abspath("my_test") + ".exp")

    def test_load(self):
        # TODO
        # self.experiment.load_fit(self.path_save + ".exp")
        pass

    def test_baseline_correction(self):
        self.experiment.baseline_substraction()
        value = self.check_preprocessing_function("baseline_substraction")
        self.assertEqual(value, True)

    def test_subtract_polynomial_baseline(self):
        wave_points = self.experiment.wavelength[::10]
        self.experiment.subtract_polynomial_baseline(wave_points)
        value = self.check_preprocessing_function("subtract_polynomial_baseline")
        self.assertEqual(value, True)

    def test_cut_time(self):
        mini, maxi = 10, 300
        self.experiment.cut_time(mini, maxi)
        value = self.check_preprocessing_function("cut_time")
        self.assertEqual(value, True)

    def test_cut_wavelength(self):
        mini, maxi = 1550, 1600
        self.experiment.cut_wavelength(mini, maxi, "cut")
        value = self.check_preprocessing_function("cut_wavelength")
        self.assertEqual(value, True)

    def test_average_time(self):
        self.experiment.average_time(50, 10)
        value = self.check_preprocessing_function("average_time")
        self.assertEqual(value, True)

    def test_derivate_data(self):
        self.experiment.derivate_data(5, 3)
        value = self.check_preprocessing_function("derivate_data")
        self.assertEqual(value, True)

    @parameterized.expand([[55, "time"],
                           [1575, "wavelength"]])
    def test_delete_points(self, point, dimension):
        self.experiment.delete_points(point, dimension)
        value = self.check_preprocessing_function("delete_points", dimension)
        self.assertEqual(value, True)

    def test_shift_time(self):
        self.experiment.shift_time(1)
        value = self.check_preprocessing_function("shift_time")
        self.assertEqual(value, True)

    @parameterized.expand([[[5, 30], 'constant', 5],
                           [[5, 30], 'constant', 8],
                           [[5, 30], 'exponential', 5]])
    def test_define_weights(self, rango, typo, val):
        self.experiment.define_weights(rango, typo, val)
        vec_res = self.experiment.weights['vector']
        self.assertEqual(self.experiment.weights['apply'], True)
        self.assertEqual(len(self.experiment.weights), 5)
        self.assertEqual(len(vec_res), len(self.experiment.x))

#                         t0, fwhm, taus, tau_inf, opt_fwhm, vary_t0, global_t0, y0
    @parameterized.expand([[0, None, [2, 8, 30], 1E12, False, True, True, None],
                           [5, 0.12, [2, 8, 30], None, True, True, True, 0],
                           [0, None, [2, 25, 90], 1E12, False, True, True, 8],
                           [0, 0.008, [5, 90], None, False, False, True, 0],
                           [0, 0.16, [5, 90], 1E12, False, True, False, None],
                           [5, 0.16, [5, 90], 1E12, False, True, False, None]])
    def test_initialize_exp_params(self, t0, fwhm, taus, tau_inf,
                                   opt_fwhm, vary_t0,
                                   global_t0, y0):

        # the next if statement is to have two cases where the correction of GVD
        # is set to True and we can verify the global_t0
        if t0 == 5:
            self.experiment.chirp_corrected = True

        if len(taus) == 3:
            self.experiment.initialize_exp_params(t0,
                                                  fwhm,
                                                  taus[0], taus[1], taus[2],
                                                  tau_inf=tau_inf,
                                                  vary_t0=vary_t0,
                                                  opt_fwhm=opt_fwhm,
                                                  global_t0=global_t0,
                                                  y0=y0)
        elif len(taus) == 2:
            self.experiment.initialize_exp_params(t0,
                                                  fwhm,
                                                  taus[0], taus[1],
                                                  tau_inf=tau_inf,
                                                  vary_t0=vary_t0,
                                                  opt_fwhm=opt_fwhm,
                                                  global_t0=global_t0,
                                                  y0=y0)

        self.assertTrue(self.experiment._params_initialized, 'Exponential')
        self.assertEqual(self.experiment.params['t0_1'].value, t0)
        self.assertEqual(self.experiment._exp_no, len(taus))
        if fwhm is not None:
            self.assertTrue(self.experiment._deconv)
            self.assertEqual(self.experiment.params['t0_1'].vary, vary_t0)
            self.assertEqual(self.experiment.params['fwhm_1'].vary, opt_fwhm)
            self.assertEqual(self.experiment.params['fwhm_1'].value, fwhm)
            self.assertEqual(self.experiment._tau_inf, tau_inf)
        else:
            self.assertFalse(self.experiment.params['t0_1'].vary)
        if y0 is not None:
            self.assertEqual(self.experiment.params["y0_1"].value, y0)

        if self.experiment.chirp_corrected:
            if global_t0:
                self.assertEqual(self.experiment.params['t0_2'].expr, 't0_1')
            else:
                self.assertEqual(self.experiment.params['t0_2'].expr, None)

        self.experiment.chirp_corrected = False


if __name__ == '__main__':
    unittest.main()
