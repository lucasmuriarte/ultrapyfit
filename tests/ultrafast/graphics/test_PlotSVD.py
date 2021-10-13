import unittest
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.graphics.PlotSVD import PlotSVD
import numpy as np
from parameterized import parameterized
import matplotlib.pyplot as plt


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
    @classmethod
    def setUpClass(self):
        path = 'examples/data/gauss_denoised.csv'
        self.time, self.data, self.wave = read_data(path, wave_is_row=True)
        self.data_select, self.wave_select = select_traces(self.data, self.wave, 'auto')
        self.svd_explorer = PlotSVD(self.time, self.data, self.wave, self.data_select, self.wave_select)
        self.full_svd_values = (np.linalg.svd(self.data, full_matrices=False, compute_uv=False)) ** 2
        self.select_svd_values = (np.linalg.svd(self.data_select, full_matrices=False, compute_uv=False)) ** 2
        

    def test_select_traces(self):
        self.svd_explorer.select_traces('all')
        self.assertTrue(assertEqualArray(self.svd_explorer.selected_traces, self.data))
        self.assertTrue(assertEqualArray(self.svd_explorer.selected_wavelength, self.wave))
        self.svd_explorer.select_traces('auto')
        self.assertTrue(assertEqualArray(self.svd_explorer.selected_traces, self.data_select))
        self.assertTrue(assertEqualArray(self.svd_explorer.selected_wavelength, self.wave_select))

    @parameterized.expand([[10],
                           [15],
                           [25]])
    def test__calculateSVD(self, val):
        u, s, v = self.svd_explorer._calculateSVD(val)
        self.assertEqual(len(s), val)

    @parameterized.expand([["all"],
                           ["select"]])
    def tes_plot_singular_values(self, select):
        fig, ax = self.svd_explorer(select)
        plotted = ax.lines[0]
        if select == "all":
            self.assertTrue(assertEqualArray(plotted, self.full_svd_values))
        else:
            self.assertTrue(assertEqualArray(plotted, self.select_svd_values))
        plt.close(fig)

    @parameterized.expand([[2, False],
                           [5, True]])
    def test_plotSVD(self, vector, select):
        self.svd_explorer.plotSVD(vector, select)
        self.assertEqual(len(self.svd_explorer._ax), 3)
        self.assertEqual(len(self.svd_explorer._ax[0].lines), vector)
        self.assertTrue((self.svd_explorer.S[:3] > 0).all())
        self.assertTrue((self.svd_explorer.S[3:] == 0).all())
        if select:
            self.assertTrue((self.svd_explorer._button_svd_select is not None))
        else:
            self.assertTrue((self.svd_explorer._button_svd_select is None))
        plt.close(self.svd_explorer._fig)
    
    @parameterized.expand([[3],
                           [7],
                           [2]])
    def test__updatePlotSVD(self, val):
        self.svd_explorer.plotSVD()
        self.svd_explorer._specSVD.val = val
        for cid, func in self.svd_explorer._specSVD.observers.items():
            func('motion_notify_event')
        self.assertEqual(len(self.svd_explorer._ax[0].lines), val)
        self.svd_explorer._specSVD.val = 5
        for cid, func in self.svd_explorer._specSVD.observers.items():
            func('motion_notify_event')
        self.assertEqual(len(self.svd_explorer._ax[0].lines), 5)
        plt.close(self.svd_explorer._fig)

    def test__selectSVD(self):
        self.svd_explorer.plotSVD(select=True)
        for cid, func in self.svd_explorer._button_svd_select.observers.items():
            func('button_press_event')
        self.assertTrue(self.svd_explorer._SVD_fit)
        self.assertEqual(self.svd_explorer.selected_traces.shape[1], 1)
        self.svd_explorer.select_traces('auto')
        plt.close(self.svd_explorer._fig)

    def test__close_svd_fig(self):
        self.svd_explorer.plotSVD()
        self.svd_explorer._close_svd_fig()
        self.assertTrue((self.svd_explorer._ax is None))
        self.assertTrue((self.svd_explorer._fig is None))
        self.assertTrue((self.svd_explorer._number_of_vectors_plot is None))
        self.assertTrue((self.svd_explorer._specSVD is None))
        self.assertTrue((self.svd_explorer._button_svd_select is None))
        self.assertTrue((self.svd_explorer.vertical_SVD is None))


if __name__ == '__main__':
    unittest.main()
