import unittest
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.graphics.PlotSVD import PlotSVD
import numpy as np
from parameterized import parameterized
import matplotlib.pyplot as plt


path = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_gauss_denoised.csv'
time, data, wave = read_data(path, wave_is_row=True)
data_select, wave_select = select_traces(data, wave, 'auto')
svd_explorer = PlotSVD(time, data, wave, data_select, wave_select)
full_svd_values = (np.linalg.svd(data, full_matrices=False, compute_uv=False)) ** 2
select_svd_values = (np.linalg.svd(data_select, full_matrices=False, compute_uv=False)) ** 2


def assertEqualArray(array1, array2):
    """
    returns "True" if all elements of two arrays are identical
    """
    if type(array1) == list:
        array1 = np.array(array1)
    if type(array2) == list:
        array2 = np.array(array2)
    value = (array1 == array2).all()
    return value


class MyTestCase(unittest.TestCase):

    def test_select_traces(self):
        svd_explorer.select_traces('all')
        self.assertTrue(assertEqualArray(svd_explorer.selected_traces, data))
        self.assertTrue(assertEqualArray(svd_explorer.selected_wavelength, wave))
        svd_explorer.select_traces('auto')
        self.assertTrue(assertEqualArray(svd_explorer.selected_traces, data_select))
        self.assertTrue(assertEqualArray(svd_explorer.selected_wavelength, wave_select))

    @parameterized.expand([[10],
                           [15],
                           [25]])
    def test__calculateSVD(self, val):
        u, s, v = svd_explorer._calculateSVD(val)
        self.assertEqual(len(s), val)

    @parameterized.expand([["all"],
                           ["select"]])
    def tes_plot_singular_values(self, select):
        fig, ax = svd_explorer(select)
        plotted = ax.lines[0]
        if select == "all":
            self.assertTrue(assertEqualArray(plotted, full_svd_values))
        else:
            self.assertTrue(assertEqualArray(plotted, select_svd_values))
        plt.close(fig)

    @parameterized.expand([[2, False],
                           [5, True]])
    def test_plotSVD(self, vector, select):
        svd_explorer.plotSVD(vector, select)
        self.assertEqual(len(svd_explorer._ax), 3)
        self.assertEqual(len(svd_explorer._ax[0].lines), vector)
        self.assertTrue((svd_explorer.S[:3] > 0).all())
        self.assertTrue((svd_explorer.S[3:] == 0).all())
        if select:
            self.assertTrue((svd_explorer._button_svd_select is not None))
        else:
            self.assertTrue((svd_explorer._button_svd_select is None))
        plt.close(svd_explorer._fig)
    
    @parameterized.expand([[3],
                           [7],
                           [2]])
    def test__updatePlotSVD(self, val):
        svd_explorer.plotSVD()
        svd_explorer._specSVD.val = val
        for cid, func in svd_explorer._specSVD.observers.items():
            func('motion_notify_event')
        self.assertEqual(len(svd_explorer._ax[0].lines), val)
        svd_explorer._specSVD.val = 5
        for cid, func in svd_explorer._specSVD.observers.items():
            func('motion_notify_event')
        self.assertEqual(len(svd_explorer._ax[0].lines), 5)
        plt.close(svd_explorer._fig)

    def test__selectSVD(self):
        svd_explorer.plotSVD(select=True)
        for cid, func in svd_explorer._button_svd_select.observers.items():
            func('button_press_event')
        self.assertTrue(svd_explorer._SVD_fit)
        self.assertEqual(svd_explorer.selected_traces.shape[1], 1)
        svd_explorer.select_traces('auto')
        plt.close(svd_explorer._fig)

    def test__close_svd_fig(self):
        svd_explorer.plotSVD()
        svd_explorer._close_svd_fig()
        self.assertTrue((svd_explorer._ax is None))
        self.assertTrue((svd_explorer._fig is None))
        self.assertTrue((svd_explorer._number_of_vectors_plot is None))
        self.assertTrue((svd_explorer._specSVD is None))
        self.assertTrue((svd_explorer._button_svd_select is None))
        self.assertTrue((svd_explorer.vertical_SVD is None))


if __name__ == '__main__':
    unittest.main()
