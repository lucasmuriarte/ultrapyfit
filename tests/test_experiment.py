import os
import io
import sys
import unittest
import numpy as np
from parameterized import parameterized
from copy import deepcopy

from ultrafast.experiment import Experiment
from ultrafast.utils.divers import get_root_directory
from ultrafast.utils.test_tools import ArrayTestCase


class TestExperiment(ArrayTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(
            get_root_directory(),
            'examples/data/exp3_data_denoised.csv'
        )

        cls.path_save = os.path.join(
            get_root_directory(),
            'tests', "test_experiment_tmp"
        )
        cls.path = os.path.abspath(path)
        cls.experiment = Experiment.load_data(
            cls.path, wave_is_row=True)

        cls.original_time = deepcopy(cls.experiment.time)

    def check_preprocessing_function(self, name, extra=None):
        action_number = len(self.experiment.action_records.__dict__) - 3
        action = getattr(self.experiment.action_records, f"_{action_number}")
        suppose_action = " ".join(name.split("_"))

        if extra is not None:
            suppose_action += f" {extra}"

        action_written = action == suppose_action
        data_added = hasattr(
            self.experiment.preprocessing.data_sets,
            "before_" + name
        )

        report_updated = hasattr(self.experiment.preprocessing.report, name)

        true_values = \
            (action_written, data_added, report_updated) == (True, True, True)

        return true_values

    def test__initialized(self):
        self.assertTrue(
            hasattr(self.experiment.preprocessing.data_sets, "original_data"))

        self.assertTrue(
            hasattr(self.experiment.fit.fit_records, "single_fits"))

        self.assertTrue(
            hasattr(self.experiment.fit.fit_records, "bootstrap_record"))

        self.assertTrue(
            hasattr(self.experiment.fit.fit_records, "conf_interval"))

        self.assertTrue(
            hasattr(self.experiment.fit.fit_records, "target_models"))

        self.assertTrue(
            hasattr(self.experiment.fit.fit_records, "global_fits"))

        self.assertTrue(
            hasattr(self.experiment.fit.fit_records, "integral_band_fits"))

        self.assertTrue(
            hasattr(self.experiment, "_unit_formater"))

    def test_time(self):
        val = (self.experiment.time == self.experiment.x).all()

        self.assertTrue(val)

    def test_time_setter(self):
        new_time = np.random.randn(1000)
        self.experiment.time = new_time
        val = (self.experiment.x == new_time).all()

        self.assertTrue(val)
        # self.experiment.x = self.original_time

    def test_load_data(self):
        experiment = Experiment.load_data(self.path, wave_is_row=True)

        self.assertTrue(type(experiment) == Experiment)

    def test_save(self):
        self.experiment.save(self.path_save)
        is_file = os.path.isfile(self.path_save + ".exp")
        os.remove(self.path_save + ".exp")
        self.assertTrue(is_file)

    def test_load(self):
        # TODO
        # self.experiment.load_fit(self.path_save + ".exp")
        ...

    def test_describe_data(self):
        path_output = os.path.join(
            get_root_directory(),
            "tests/resources/describe_data_test_output.txt")

        # TODO make it utf-8
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
        path_output = os.path.join(
            get_root_directory(),
            "tests/resources/print_fit_results_test_output.txt")

        # TODO make it utf-8
        with open(path_output) as f:
            lines = f.readlines()
            expected_output = "".join(lines).replace("NÂ", "N")

        experiment = Experiment.load_data(self.path, wave_is_row=True)
        experiment.select_traces()
        experiment.fit.initialize_exp_params(0, None, 5, 20, 300)
        experiment.fit.global_fit()

        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        experiment.fit.print_results()  # Call function.
        output = captured_output.getvalue()

        self.assertEqual(expected_output, output)

    def test_general_report(self):
        path_output = os.path.join(
            get_root_directory(),
            "tests/resources/general_report_test_output.txt")

        with open(path_output) as f:
            lines = f.readlines()
            expected_output = "".join(lines).replace("NÂ", "N")

        experiment = Experiment.load_data(self.path, wave_is_row=True)
        experiment.select_traces()
        experiment.fit.initialize_exp_params(0, None, 5, 20, 300)
        experiment.fit.global_fit()
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
        self.experiment.preprocessing.baseline_substraction()
        value = self.check_preprocessing_function("baseline_substraction")
        self.assertTrue(value)

    def test_subtract_polynomial_baseline(self):
        wave_points = self.experiment.wavelength[::10]
        self.experiment.preprocessing.subtract_polynomial_baseline(wave_points)
        
        self.assertTrue(self.check_preprocessing_function(
            "subtract_polynomial_baseline"))

    def test_cut_time(self):
        mini, maxi = 10, 300
        self.experiment.preprocessing.cut_time(mini, maxi)
        value = self.check_preprocessing_function("cut_time")

        self.assertTrue(value)

    def test_cut_wavelength(self):
        mini, maxi = 1550, 1600
        self.experiment.preprocessing.cut_wavelength(mini, maxi, "cut")
        value = self.check_preprocessing_function("cut_wavelength")

        self.assertTrue(value)

    def test_average_time(self):
        self.experiment.preprocessing.average_time(50, 10)
        value = self.check_preprocessing_function("average_time")

        self.assertTrue(value)

    def test_derivate_data(self):
        self.experiment.preprocessing.derivate_data(5, 3)
        value = self.check_preprocessing_function("derivate_data")

        self.assertTrue(value)

    def test_calibrate_wavelength(self):
        original_wavelengths = self.experiment.wavelength

        values = [(a * 100, b) for a, b in
                  enumerate(self.experiment.wavelength[::100])]

        pixels = [i[0] for i in values]
        waves = [i[1] for i in values]
        pix = np.array([a for a, b in enumerate(self.experiment.wavelength)])

        self.experiment.wavelength = pix
        self.experiment.preprocessing.calibrate_wavelength(pixels, waves)

        value = self.check_preprocessing_function("calibrate_wavelength")

        self.assertNearlyEqualArray(
            self.experiment.wavelength,
            original_wavelengths, 10)

        self.assertTrue(value)

    @parameterized.expand([
        [55, "time"],
        [1575, "wavelength"]])
    def test_delete_points(self, point, dimension):
        self.experiment.preprocessing.delete_points(point, dimension)
        value = self.check_preprocessing_function("delete_points", dimension)

        self.assertTrue(value)

    def test_shift_time(self):
        self.experiment.preprocessing.shift_time(1)
        value = self.check_preprocessing_function("shift_time")

        self.assertTrue(value)

    @parameterized.expand([
        [[5, 30], 'constant', 5],
        [[5, 30], 'constant', 8],
        [[5, 30], 'exponential', 5]])
    def test_define_weights(self, rango, typo, val):
        self.experiment.fit.define_weights(rango, typo, val)
        vec_res = self.experiment.fit.weights['vector']

        self.assertTrue(self.experiment.fit.weights['apply'])
        self.assertEqual(len(self.experiment.fit.weights), 5)
        self.assertEqual(len(vec_res), len(self.experiment.x))

    @parameterized.expand([
        [0, None, [2, 8, 30], 1E12, False, True, True, None],
        [5, 0.12, [2, 8, 30], None, True, True, True, 0],
        [0, None, [2, 25, 90], 1E12, False, True, True, 8],
        [0, 0.008, [5, 90], None, False, False, True, 0],
        [0, 0.16, [5, 90], 1E12, False, True, False, None],
        [5, 0.16, [5, 90], 1E12, False, True, False, None]])
    def test_initialize_exp_params(self, t0, fwhm, taus, tau_inf,
                                   opt_fwhm, vary_t0, global_t0, y0):
        # the next "if" statement is to have two cases where the correction of
        # GVD is set to True to verify the global_t0 parameter
        if t0 == 5:
            self.experiment.preprocessing.chirp_corrected = True

        if len(taus) == 3:
            self.experiment.fit.initialize_exp_params(
                t0,
                fwhm,
                taus[0],
                taus[1],
                taus[2],
                tau_inf=tau_inf,
                vary_t0=vary_t0,
                opt_fwhm=opt_fwhm,
                global_t0=global_t0,
                y0=y0
            )

        elif len(taus) == 2:
            self.experiment.fit.initialize_exp_params(
                t0,
                fwhm,
                taus[0],
                taus[1],
                tau_inf=tau_inf,
                vary_t0=vary_t0,
                opt_fwhm=opt_fwhm,
                global_t0=global_t0,
                y0=y0
            )

        self.assertEqual(
            self.experiment.fit._params_initialized, 'Exponential')

        self.assertEqual(
            self.experiment.fit.params['t0_1'].value, t0)

        self.assertEqual(
            self.experiment.fit._exp_no, len(taus))

        if fwhm is not None:
            self.assertTrue(self.experiment.fit._deconv)

            self.assertEqual(
                self.experiment.fit.params['t0_1'].vary, vary_t0)

            self.assertEqual(
                self.experiment.fit.params['fwhm_1'].vary, opt_fwhm)

            self.assertEqual(
                self.experiment.fit.params['fwhm_1'].value, fwhm)

            self.assertEqual(
                self.experiment.fit._tau_inf, tau_inf)

        else:
            self.assertFalse(self.experiment.fit.params['t0_1'].vary)

        if y0 is not None:
            self.assertEqual(self.experiment.fit.params["y0_1"].value, y0)

        if self.experiment.preprocessing.chirp_corrected:
            if global_t0:
                self.assertEqual(
                    self.experiment.fit.params['t0_2'].expr, 't0_1')

            else:
                self.assertEqual(
                    self.experiment.fit.params['t0_2'].expr, None)

        self.experiment.chirp_corrected = False

    def test_initialized_target_params(self):
        # TODO
        ...

    def test_undo_last_preprocesing(self):
        experiment1 = Experiment.load_data(self.path, wave_is_row=True)
        experiment2 = Experiment.load_data(self.path, wave_is_row=True)
        experiment2.preprocessing.baseline_substraction()
        different = (experiment1.data != experiment2.data).all()

        self.assertTrue(different)

        experiment2.preprocessing.undo_last_preprocesing()
        equal = (experiment1.data == experiment2.data).all()

        self.assertTrue(equal)

    def test_global_fit(self):
        self.experiment.select_traces()
        self.experiment.fit.initialize_exp_params(0, None, 5, 20, 300)
        self.experiment.fit.global_fit()

        recovered_times = [self.experiment.fit.params["tau1_1"].value,
                           self.experiment.fit.params["tau2_1"].value,
                           self.experiment.fit.params["tau3_1"].value]


        self.assertNearlyEqualArray(recovered_times, [8, 30, 200], decimal=10)

        self.assertEqual(
            self.experiment.fit._fit_number, 1)

        self.assertEqual(
            len(self.experiment.fit.fit_records.global_fits), 1)

    def test_single_exp_fit(self):
        self.experiment.fit.single_exp_fit(
            1480, 1, 0, None, 0.7, 40, plot=False)

        self.assertEqual(len(self.experiment.fit.fit_records.single_fits), 1)

        tau1 = self.experiment.fit.fit_records.single_fits[1].params["tau1_1"]
        tau2 = self.experiment.fit.fit_records.single_fits[1].params["tau2_1"]
        self.assertEqual(round(tau1.value), 1)
        self.assertEqual(round(tau2.value), 8)

    @parameterized.expand([
        ["baseline_substraction"],
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
            experiment.preprocessing.restore_data(action)

        except Exception as error:
            exception_raise = True
            message = error.msg

        self.assertTrue(exception_raise)
        self.assertEqual(f"data has not been {action}", message)

        if action == "baseline_substraction":
            experiment.preprocessing.baseline_substraction()

        elif action == "delete_points":
            experiment.preprocessing.delete_points(55, "time")
            experiment.preprocessing.delete_points(1575, "wavelength")

        elif action == "subtract_polynomial_baseline":
            wave_points = experiment.wavelength[::10]
            experiment.preprocessing.subtract_polynomial_baseline(wave_points)

        elif action == "calibrate_wavelength":
            experiment.preprocessing.calibrate_wavelength(
                [1500, 1600, 1650],
                [1, 200, 300])

        elif action == "shift_time":
            experiment.preprocessing.shift_time(1)

        elif action == "derivate_data":
            experiment.preprocessing.derivate_data(5, 3)

        elif action == "cut_wavelength":
            experiment.preprocessing.cut_wavelength(1550, 1600, "cut")

        elif action == "cut_time":
            experiment.preprocessing.cut_time(10, 300)

        elif action == "average_time":
            experiment.preprocessing.average_time(50, 10)

        experiment.preprocessing.restore_data(action)
        container = getattr(
            experiment.preprocessing.data_sets, f"before_{action}")
            
        self.assertEqualArray(experiment.data, container.data)
        self.assertEqualArray(experiment.x, container.x)
        self.assertEqualArray(experiment.wavelength, container.wavelength)


if __name__ == '__main__':
    unittest.main()
