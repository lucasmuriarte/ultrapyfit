import unittest
from unittest import mock
from ultrafast.outils import readData
from ultrafast.PlotTransientClass import ExploreData
from ultrafast.outils import select_traces
import numpy as np
from parameterized import parameterized 
import matplotlib as mpl
import matplotlib.pyplot as plt


path = 'C:/Users/lucas/git project/chempyspec/examples/3_exp_data_gauss_denoised.csv'
time, data, wave = readData(path, wave_is_row=True)
data_select, wave_select = select_traces(data, wave, 'auto')
explorer = ExploreData(time, data, wave, data_select, wave_select)
explorer.units['wavelength_unit'] = 'nm'


@mock.patch("%s.my_module.plt" % __name__)
def test_module(mock_plt):
    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    my_module.plot_data(x, y, "my title")

    # Assert plt.title has been called with expected arg
    mock_plt.title.assert_called_once_with("my title")

    # Assert plt.figure got called
    assert mock_plt.figure.called


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


class TestPlotTransientClass(unittest.TestCase):

    @parameterized.expand([[True],
                           [False]])
    def test_plot_trace(self, auto):
        fig, ax = explorer.plot_traces(auto)
        lines = ax.lines
        if auto:
            self.assertEqual(len(lines), 11)
            for i in range(len(lines) - 1):
                 self.assertTrue(assertEqualArray(data[:, i*10], lines[i]._y))
        else:
            self.assertEqual(len(lines), data_select.shape[1] + 1)
            for i in range(len(lines) - 1):
                 self.assertTrue(assertEqualArray(data_select[:, i], lines[i]._y))
        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()
        self.assertEqual(x_l, 'Time (ps)')
        self.assertEqual(y_l, '$\Delta$A')
        legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]
        if len(lines) <= 11:
            self.assertTrue(len(legends) == 1)
        else:
            self.assertTrue(len(legends) == 0)
        plt.close(fig)

    def test_plot_spectra_all(self):
        fig, ax = explorer.plot_spectra()
        lines = ax.lines
        self.assertEqual(len(lines), len(time)+1)
        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()
        self.assertEqual(x_l, 'Wavelength (nm)')
        self.assertEqual(y_l, '$\Delta$A')
        legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]
        self.assertTrue(len(legends) == 0)
        for i in range(len(time)):
            self.assertTrue(assertEqualArray(data[i, :], lines[i]._y))
        plt.close(fig)

    @parameterized.expand([[[-2, -0.5, 0, 0.5, 1, 15, 20, 100, 150], True],
                           [[-2, -0.5, 0, 0.5, 1, 15, 20, 100, 150], False],
                           [[-2, 150], True]])
    def test_plot_spectra_list_times(self, lista, legend):
        fig, ax = explorer.plot_spectra(lista, legend=legend)
        lines = ax.lines
        self.assertEqual(len(lista) + 1, len(lines))
        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()
        self.assertEqual(x_l, 'Wavelength (nm)')
        self.assertEqual(y_l, '$\Delta$A')
        index = [np.argmin(abs(time - i)) for i in lista]
        for i in range(len(lista)):
            self.assertTrue(assertEqualArray(data[index[i], :], lines[i]._y))
        legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]
        if legend:
            self.assertTrue(len(legends) == 1)
        else:
            self.assertTrue(len(legends) == 0)
        plt.close(fig)
    
    @parameterized.expand([[["auto", 8], None, False, False],
                           [["auto", 8,302], None, True, False],
                           [["auto", 6], [10,500], True, True],
                           [["auto", 10], None, True, False]])
    def test_plot_spectra_list_auto(self, times, rango, from_max_to_min, include_rango_max):
        fig, ax = explorer.plot_spectra(times, rango=rango, from_max_to_min=from_max_to_min, 
                                        include_rango_max=include_rango_max)
        lines = ax.lines
        self.assertTrue(len(lines) == times[1] + 1)
        leg = ax.legend_.get_texts()
        if from_max_to_min and rango is None:
            if len(times) == 3:
                idx = np.argmax(abs(data[:, 83]))
                self.assertEqual(leg[0].get_text(), '{:.2f}'.format(time[idx]) + ' ps')
            else:
                idx = np.unravel_index(np.argmax(abs(data), axis=None), data.shape)
                self.assertEqual(leg[0].get_text(), '{:.2f}'.format(time[idx[0]]) + ' ps')
        if include_rango_max:
            self.assertEqual(leg[-1].get_text(), '496.82 ps')
        plt.close(fig)

    def test__getWaveLabel(self):
        text = explorer._get_wave_label()
        self.assertEqual(text, 'Wavelength (nm)')
        explorer.units['wavelength_unit'] = 'cm-1'
        text = explorer._get_wave_label()
        self.assertEqual(text, 'Wavenumber (cm$^{-1}$)')
        explorer.units['wavelength_unit'] = 'nm'
        
    @parameterized.expand([[["auto", 8], None, False, False],
                           [["auto", 8, 302], None, True, False],
                           [["auto", 6], [10,500], True, True],
                           [["auto", 10], None, True, False],
                           [[-2, -0.5, 0, 0.5, 1, 15, 20, 100, 150], None, False, False],
                           ["all", None, False, False]])
    def test__time_to_real_times(self, times, rango, include_max, from_max_to_min):
        result = explorer._time_to_real_times(times, rango, include_max, from_max_to_min)
        if type(times) == list:
            if times[0] == 'auto':
                wave = None if len(times) == 2 else times[2]
                output = explorer._get_auto_points(times[1], wave, rango, include_max, from_max_to_min)
            else:
                output = time[[np.argmin(abs(time-i)) for i in times]]
        else:
            output = time
        self.assertTrue(assertEqualArray(result, output))
     
    @parameterized.expand([[["auto", 8], None, False, False],
                           [["auto", 8, 302], None, True, False],
                           [["auto", 6], [10,500], True, True],
                           [["auto", 10], None, True, False]])
    def test__get_auto_points(self, times, rango, include_max, from_max_to_min):
        wave_p = None if len(times) == 2 else times[2]
        maxi = 83 if wave_p is None else np.argmin(abs(wave-wave_p))
        output = explorer._get_auto_points(times[1], wave_p, rango, include_max, from_max_to_min)
        idx = [np.argmin(abs(time-i)) for i in explorer._get_auto_points()]
        differences = np.array([data[i, maxi] - data[i+1, maxi] for i in idx[:-1]])
        self.assertEqual(times[1], len(output))
        self.assertTrue((differences < 1e-4).all())

    def test_module(self):
        mock_plt = mock.Mock(ExploreData(time, data, wave, data_select, wave_select))
        mock_plt.plot3D()
        mock_plt.plot3D.assert_called()

    
if __name__ == '__main__':
    unittest.main()
