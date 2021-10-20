import unittest
import unittest.mock
from ultrafast.experiment import Experiment
import os
import numpy as np
from copy import deepcopy
from parameterized import parameterized
import io
import sys


class TestExperiment(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExperiment, self).__init__(*args, **kwargs)
        path = '../../examples/data/exp3_data_denoised.csv'
        self.path_save = os.path.abspath("my_test")
        self.path = os.path.abspath(path)
        self.experiment = Experiment.load_data(self.path, wave_is_row=True)
        self.original_time = deepcopy(self.experiment.time)

    def assertEqualArray(self, array1, array2):
        """
        returns "True" if all elements of two arrays are identical
        """
        value = (array1 == array2).all()
        return value

    def assertNearlyEqualArray(self, array1, array2, decimal):
        """
        returns "True" if all elements of two arrays
        are identical until decimal
        """
        dif = np.array(array1) - np.array(array2)
        value = (dif < 10 ** (-decimal)).all()
        return value

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
        data_added = hasattr(self.experiment.data_sets, "before_" + name)
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
        self.assertEqual(
            hasattr(self.experiment.fit_records, "bootstrap_record"),
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

    def test_load_data(self):
        experiment = Experiment.load_data(self.path, wave_is_row=True)
        self.assertTrue(type(experiment) == Experiment)

    def test_save(self):
        self.experiment.save(self.path_save)
        is_file = os.path.isfile(self.path_save + ".exp")
        self.assertEqual(is_file, True)
        os.remove(os.path.abspath("my_test") + ".exp")

    def test_load(self):
        # TODO
        # self.experiment.load_fit(self.path_save + ".exp")
        pass

    def test_describe_data(self):
        path_output = "resources/describe_data_test_output.txt"
        path_output = os.path.abspath(path_output)
        with open(path_output) as f:
            lines = f.readlines()
            expected_output = "".join(lines).replace("NÂ", "N")
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.experiment.describe_data()  # Call function.
        # sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertEqual(expected_output, output)

    def test_print_results(self):
        path_output = "resources/print_fit_results_test_output.txt"
        path_output = os.path.abspath(path_output)
        with open(path_output) as f:
            lines = f.readlines()
            expected_output = "".join(lines).replace("NÂ", "N")
        experiment = Experiment.load_data(self.path, wave_is_row=True)
        experiment.select_traces()
        experiment.initialize_exp_params(0, None, 5, 20, 300)
        experiment.global_fit()
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        experiment.print_results()  # Call function.
        output = captured_output.getvalue()
        self.assertEqual(expected_output, output)

    def test_general_report(self):
        path_output = "resources/general_report_test_output.txt"
        path_output = os.path.abspath(path_output)
        with open(path_output) as f:
            lines = f.readlines()
            expected_output = "".join(lines).replace("NÂ", "N")
        experiment = Experiment.load_data(self.path, wave_is_row=True)
        experiment.select_traces()
        experiment.initialize_exp_params(0, None, 5, 20, 300)
        experiment.global_fit()
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        experiment.general_report()  # Call function.
        output = captured_output.getvalue()

        # the middle part of the report changes every time is run since the
        # hour is printed out and other parameter are different
        # thus we verify only that the initial and end are idential
        self.assertEqual(expected_output[0:30], output[0:30])
        self.assertEqual(expected_output[-30:], output[-30:])

    def test_baseline_correction(self):
        self.experiment.baseline_substraction()
        value = self.check_preprocessing_function("baseline_substraction")
        self.assertEqual(value, True)

    def test_subtract_polynomial_baseline(self):
        wave_points = self.experiment.wavelength[::10]
        self.experiment.subtract_polynomial_baseline(wave_points)
        value = self.check_preprocessing_function(
            "subtract_polynomial_baseline")
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

    def test_calibrate_wavelength(self):
        original_wavelengths = self.experiment.wavelength
        values = [(a * 100, b) for a, b in
                  enumerate(self.experiment.wavelength[::100])]
        pixels = [i[0] for i in values]
        waves = [i[1] for i in values]
        pix = np.array([a for a, b in enumerate(self.experiment.wavelength)])
        self.experiment.wavelength = pix
        self.experiment.calibrate_wavelength(pixels, waves)
        value = self.check_preprocessing_function("calibrate_wavelength")
        equal = self.assertNearlyEqualArray(self.experiment.wavelength,
                                            original_wavelengths, 10)
        self.assertTrue(equal)
        self.assertTrue(value)

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

    #                     t0, fwhm, taus, tau_inf, opt_fwhm, vary_t0, global_t0, y0
    @parameterized.expand([[0, None, [2, 8, 30], 1E12, False, True, True, None],
                           [5, 0.12, [2, 8, 30], None, True, True, True, 0],
                           [0, None, [2, 25, 90], 1E12, False, True, True, 8],
                           [0, 0.008, [5, 90], None, False, False, True, 0],
                           [0, 0.16, [5, 90], 1E12, False, True, False, None],
                           [5, 0.16, [5, 90], 1E12, False, True, False, None]])
    def test_initialize_exp_params(self, t0, fwhm, taus, tau_inf,
                                   opt_fwhm, vary_t0,
                                   global_t0, y0):

        # the next "if" statement is to have two cases where the correction of
        # GVD is set to True ato verify the global_t0 parameter
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

    def test_initialized_target_params(self):
        # TODO
        pass

    def test_undo_last_preprocesing(self):
        experiment1 = Experiment.load_data(self.path, wave_is_row=True)
        experiment2 = Experiment.load_data(self.path, wave_is_row=True)
        experiment2.baseline_substraction()
        different = (experiment1.data != experiment2.data).all()
        self.assertTrue(different)
        experiment2.undo_last_preprocesing()
        equal = (experiment1.data == experiment2.data).all()
        self.assertTrue(equal)

    def test_global_fit(self):
        self.experiment.select_traces()
        self.experiment.initialize_exp_params(0, None, 5, 20, 300)
        self.experiment.global_fit()
        recovered_times = [self.experiment.params["tau1_1"].value,
                           self.experiment.params["tau2_1"].value,
                           self.experiment.params["tau3_1"].value]
        self.assertNearlyEqualArray(recovered_times, [8, 30, 200], decimal=8)
        self.assertTrue(self.experiment._fit_number == 1)
        self.assertTrue(len(self.experiment.fit_records.global_fits) == 1)

    @parameterized.expand([["baseline_substraction"],
                           ["average_time"],
                           ["cut_time"],
                           ["cut_wavelength"],
                           ["delete_points"],
                           ["derivate_data"],
                           ["shift_time"],
                           ["subtract_polynomial_baseline"],
                           ["calibrate_wavelength"]])
    def test_restore_data(self, action):
        experiment = Experiment.load_data(self.path, wave_is_row=True)
        exception_raise = False
        message = None
        try:
            experiment.restore_data(action)
        except Exception as error:
            exception_raise = True
            message = error.msg
        self.assertTrue(exception_raise)
        self.assertEqual(f"data has not been {action}", message)
        if action == "baseline_substraction":
            experiment.baseline_substraction()
        elif action == "delete_points":
            experiment.delete_points(55, "time")
            experiment.delete_points(1575, "wavelength")
        elif action == "subtract_polynomial_baseline":
            wave_points = experiment.wavelength[::10]
            experiment.subtract_polynomial_baseline(wave_points)
        elif action == "calibrate_wavelength":
            experiment.calibrate_wavelength([1500, 1600, 1650], [1, 200, 300])
        elif action == "shift_time":
            experiment.shift_time(1)
        elif action == "derivate_data":
            experiment.derivate_data(5, 3)
        elif action == "cut_wavelength":
            experiment.cut_wavelength(1550, 1600, "cut")
        elif action == "cut_time":
            experiment.cut_time(10, 300)
        elif action == "average_time":
            experiment.average_time(50, 10)
        experiment.restore_data(action)
        container = getattr(experiment.data_sets, f"before_{action}")
        equal_data = self.assertEqualArray(experiment.data, container.data)
        self.assertTrue(equal_data)
        equal_time = self.assertEqualArray(experiment.x, container.x)
        self.assertTrue(equal_time)
        equal_wave = self.assertEqualArray(experiment.wavelength,
                                           container.wavelength)
        self.assertTrue(equal_wave)


if __name__ == '__main__':
    unittest.main()
