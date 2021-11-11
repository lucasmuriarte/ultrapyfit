import unittest
import os
import pickle
import numpy as np
from parameterized import parameterized
from unittest import mock
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.utils.test_tools import ArrayTestCase
from ultrafast.utils.divers import \
    get_root_directory, \
    read_data, \
    select_traces


class TestExploreResultsClass(ArrayTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(
            get_root_directory(),
            'examples/data/denoised_2.csv'
        )

        cls.original_taus = [8, 30, 200]

        cls.time, cls.data, cls.wave = read_data(path, wave_is_row=True)
        cls.data_select, cls.wave_select = select_traces(
            cls.data, cls.wave, 'auto')

        file = os.path.join(
            get_root_directory(),
            'examples/data/denoised_2.res'
        )

        cls.res = pickle.load(open(file, 'rb'))

        cls.result = ExploreResults(cls.res)

    def test_print_results(self):
        mock_plt = mock.Mock(ExploreResults(self.res))
        mock_plt.print_results()
        mock_plt.print_results.assert_called()

    @parameterized.expand([[1], [None]])
    def test_results(self, fit_number):
        dat = self.result.results(fit_number)

        self.assertNearlyEqualArray(self.data_select, dat, 11)

    @parameterized.expand([[1], [None]])
    def test_DAS(self, fit_number):
        dat = self.result.get_DAS(fit_number)
        row, _ = dat.shape

        self.assertEqual(len(dat), len(self.original_taus))
        self.assertEqual(len(row), self.data_select.shape[1])

    @parameterized.expand([[1], [None]])
    def test_plot_DAS(self, fit_number):
        fig, ax = self.result.plot_DAS(fit_number)
        lines = ax.lines

        self.assertEqual(len(lines), len(self.original_taus) + 1)

        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()

        self.assertEqual(x_l, 'Wavelength (nm)')
        self.assertEqual(y_l, '$\Delta$A')

        for i in range(len(lines) - 1):
            self.assertNearlyEqualArray(lines[i]._y, self.data_select[1, :], 5)

    @parameterized.expand([[1], [None]])
    def test_plot_fit(self, fit_number):
        _, ax = self.result.plot_DAS(fit_number)
        scatter = ax[1].collections
        lines = ax[1].collections
        scatter_res = ax[0].collections
        line_res = ax[0].collections

        self.assertEqual(len(scatter), len(scatter_res))
        self.assertEqual(len(line_res), 1)
        self.assertEqual(len(line_res), self.data_select.shape[1] + 1)

        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()

        self.assertEqual(x_l, 'Time (ps)')
        self.assertEqual(y_l, '$\Delta$A')

        for i in range(len(lines) - 1):
            self.assertNearlyEqualArray(
                scatter[i]._offsets.data[:, 1] - lines[i]._y,
                scatter_res[i]._offsets.data[:, 1],
                5
            )

            self.assertNearlyEqualArray(
                scatter[i]._offsets.data[:, 1],
                self.data_select[:, i],
                5
            )

    def test__legend_plot_DAS(self):
        leg = ['8.00 ps', '30.00 ps', '200.00 ps']
        legenda = self.result._legend_plot_DAS()

        for i in range(3):
            self.assertEqual(leg[i], legenda[i])


if __name__ == '__main__':
    unittest.main()
