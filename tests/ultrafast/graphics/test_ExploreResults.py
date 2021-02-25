import unittest
from unittest import mock
from ultrafast.graphics.ExploreResults import ExploreResults
import pickle
from parameterized import parameterized
from ultrafast.utils.divers import read_data, select_traces
import numpy as np


path = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_denoised_2.csv'

original_taus = [8, 30, 200]

time, data, wave = read_data(path, wave_is_row=True)
data_select, wave_select = select_traces(data, wave, 'auto')

file = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_denoised_2_results.res'
res = pickle.load(file)

result = ExploreResults(res)


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


class TestExploreResultsClass(unittest.TestCase):

    def test_print_results(self):
        mock_plt = mock.Mock(ExploreResults(res))
        mock_plt.print_results()
        mock_plt.print_results.assert_called()

    @parameterized.expand([[1],
                           [None]])
    def test_results(self, fit_number):
        dat = result.results(fit_number)
        self.assertTrue(assertNearlyEqualArray(data_select, dat, 11))

    @parameterized.expand([[1],
                           [None]])
    def test_DAS(self, fit_number):
        dat = result.DAS(fit_number)
        row, col = dat.shape
        self.assertEqual(len(dat), len(original_taus))
        self.assertEqual(len(row), data_select.shape[1])

    @parameterized.expand([[1],
                           [None]])
    def test_plot_DAS(self, fit_number):
        fig, ax = result.plot_DAS(fit_number)
        lines = ax.lines
        self.assertEqual(len(lines), len(original_taus) + 1)
        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()
        self.assertEqual(x_l, 'Wavelength (nm)')
        self.assertEqual(y_l, '$\Delta$A')
        for i in range(len(lines)-1):
            assertNearlyEqualArray(lines[i]._y, data_select[1, :], 5)

    @parameterized.expand([[1],
                           [None]])
    def test_plot_fit(self, fit_number):
        fig, ax = result.plot_DAS(fit_number)
        scatter = ax[1].collections
        lines = ax[1].collections
        scatter_res = ax[0].collections
        line_res = ax[0].collections
        self.assertEqual(len(scatter), len(scatter_res))
        self.assertEqual(len(line_res), 1)
        self.assertEqual(len(line_res), data_select.shape[1] + 1)
        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()
        self.assertEqual(x_l, 'Time (ps)')
        self.assertEqual(y_l, '$\Delta$A')
        for i in range(len(lines) - 1):
            assertNearlyEqualArray(scatter[i]._offsets.data[:, 1] - lines[i]._y,
                                   scatter_res[i]._offsets.data[:, 1], 5)
            assertNearlyEqualArray(scatter[i]._offsets.data[:, 1], data_select[:, i], 5)

    def test__legend_plot_DAS(self):
        leg = ['8.00 ps', '30.00 ps', '200.00 ps']
        legenda = result._legend_plot_DAS()
        for i in range(3):
            self.assertEqual(leg[i], legenda[i])

if __name__ == '__main__':
    unittest.main()
