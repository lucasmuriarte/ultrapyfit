# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:29:27 2020

@author: lucas
"""
from scipy.sparse.linalg import svds as svd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from ultrafast.tools import select_traces
from copy import copy


class PlotSVD:
    """
    Class to do a singular value decomposition (SVD) of the data and explore the results.

    Attributes
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2darray
        array containing the data, the number of rows should be equal to the len(x)

    wavelength: 1darray
        wavelength vector

    selected_traces: 2darray
        sub dataset of data

    selected_wavelength: 1darray
        sub dataset of wavelength

    S:
        Number of singular values calculated.

    U:
        Number of left singular vectors calculated  .

    V:
        Number of right singular vectors calculated.
    """

    def __init__(
            self,
            x,
            data,
            wavelength,
            selected_traces=None,
            selected_wavelength=None):
        self.data = data
        self.x = x
        self.wavelength = wavelength
        self.selected_traces = selected_traces
        self.selected_wavelength = selected_wavelength
        self._SVD_fit = False
        self.S = None
        self.U = None
        self.V = None
        self._fig = None
        self._ax = None
        self._number_of_vectors_plot = None
        self._specSVD = None
        self._button_svd_select = None
        self.vertical_SVD = None

    def _calculateSVD(self, vectors=15):
        """
        Calculated a truncated SVD of the data matrix using scipy.sparse.linalg.svd function

        Parameters
        ----------
        vectors:
            number of singular values and singular vectors to be calculated
        """
        u, s, v = svd(self.data, k=vectors)
        return u[:, ::-1], s[::-1], v[::-1, :]

    def plotSVD(self, vectors=1, select=False, calculate=15):
        if self._fig is not None:
            self._close_svd_fig()
        wavelength = self.wavelength
        if self.S is None or len(self.S) != calculate:
            self.U, self.S, self.V = self._calculateSVD(vectors=15)
        assert 0 < vectors < len(
            self.S), 'vector value should be between 1 and the number of calculated values'
        if vectors == 'all':
            vectors = len(self.S)
        self._fig, self._ax = plt.subplots(1, 3, figsize=(14, 6))
        self._ax[1].plot(range(1, len(self.S) + 1), self.S, marker='o')
        for i in range(vectors):
            self._ax[0].plot(self.x, self.U[:, i])
            self._ax[2].plot(wavelength, self.V[i, :])
        self._ax[0].set_title('Left singular vectors')
        self._ax[1].set_title('Eingen values')
        self._ax[2].set_title('Right singular vectors')
        self._number_of_vectors_plot = vectors
        self.vertical_SVD = self._ax[1].axvline(
            vectors, alpha=0.5, color='red', zorder=np.inf)
        axspec = self._fig.add_axes(
            [0.20, .02, 0.60, 0.01], facecolor='orange')
        self._specSVD = Slider(
            axspec, 'curve number', 1, len(
                self.S), valstep=1, valinit=vectors)
        self._specSVD.on_changed(self._updatePlotSVD)
        # self._fig.canvas.mpl_connect('close_event', self._close_svd_fig())
        if select:
            b_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
            self._button_svd_select = Button(
                b_ax, 'Select', color='tab:red', hovercolor='0.975')
            self._button_svd_select.on_clicked(self._selectSVD)
        self._fig.show()

    def _close_svd_fig(self):
        """
        reestablish initial values after closing the svd plot. This is important since this object
        cannot be pickled
        """
        self._fig = None
        self._ax = None
        self._number_of_vectors_plot = None
        self._specSVD = None
        self._button_svd_select = None
        self.vertical_SVD = None

    def plot_singular_values(self, data='all', size=14, log_scale=True):
        """
        Plot the singular values of either the whole data set or the selected data set.

        Parameters
        ----------
        data: str "all" or "select"
            If "all" the singular values plotted correspond to the whole data set
            if select correspond to the singular values of the selected traces

        size: int (default 14)
            size of the figure text labels including tick labels axis labels and legend

        log_scale: bool (default True)
            defines the scale of the y axis, if true a logarithmic scale will be set
        """

        if data == 'selected':
            dat = self.selected_traces
        else:
            dat = self.data
        svd_values = (
            np.linalg.svd(
                dat,
                full_matrices=False,
                compute_uv=False)) ** 2
        x = np.linspace(1, len(svd_values), len(svd_values))
        f, ax = plt.subplots(1)
        plt.plot(x, svd_values, marker='o', alpha=0.6, ms=4, ls='')
        plt.ylabel('Eigen values', size=size)
        plt.xlabel('number', size=size)
        plt.minorticks_on()
        ax.tick_params(
            which='both',
            direction='in',
            top=True,
            right=True,
            labelsize=size)
        if log_scale:
            plt.yscale("log")
        return f, ax

    def _updatePlotSVD(self, val):
        """
        Function to update the SVD plot with the specSVD slider
        """
        wavelength = self.wavelength
        value = int(round(self._specSVD.val))
        colores = [
            'tab:orange',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan',
            'tab:blue']
        if value > self._number_of_vectors_plot:
            if value + 1 == self._number_of_vectors_plot:
                value_c = value * 1.0
                if value > 10:
                    value_c = value - 10 * (value // 10)
                self._ax[0].plot(self.x, self.U[:, value],
                                 color=colores[value_c + 1])
                self._ax[2].plot(wavelength, self.V[value, :],
                                 color=colores[value_c + 1])
            else:
                for i in range(int(self._number_of_vectors_plot), int(value)):
                    value_c = i
                    if i > 10:
                        value_c = i - 10 * (value // 10)
                    self._ax[0].plot(self.x, self.U[:, i],
                                     color=colores[value_c - 1])
                    self._ax[2].plot(wavelength, self.V[i, :],
                                     color=colores[value_c - 1])
            self.vertical_SVD.remove()
            self.vertical_SVD = self._ax[1].axvline(
                value, alpha=0.5, color='red', zorder=np.inf)
            self._number_of_vectors_plot = value * 1.0
        elif value < self._number_of_vectors_plot:
            self.vertical_SVD.remove()
            self.vertical_SVD = self._ax[1].axvline(
                value, alpha=0.5, color='red', zorder=np.inf)
            for i in range(int(value), int(self._number_of_vectors_plot)):
                del self._ax[0].lines[-1]
                del self._ax[2].lines[-1]
            self._number_of_vectors_plot = value * 1.0
        else:
            pass
        self._fig.canvas.draw_idle()

    def _selectSVD(self, val):
        """
        function to select the left singular vectors as selected traces. This allows to
        perform a fit to the singular vector which is much faster than a global fit, since
        the number of singular vector is lower.
        """
        value = int(round(self._specSVD.val))
        self.selected_traces = self.U[:, :value]
        self.selected_wavelength = np.linspace(1, value, value)
        self._SVD_fit = True
        plt.close(self._fig)
        self._close_svd_fig()

    def select_traces(self, points=10, average=1, avoid_regions=None):
        """
        Method to select traces from the data attribute ande defines a subset of traces
        in the selected_traces attribute

        Parameters
        ----------
        points: int or list or "auto" (default 10)
            If type(space) =int: a series of traces separated by the value indicated
            will be selected.
            If type(space) = list: the traces in the list will be selected.
            If space = auto, the number of returned traces is 10 and equally spaced
            along the wavelength vector and points is set to 0

        average: int (default 1)
            Binning points surrounding the selected wavelengths.
            e. g.: if point is 1 trace = mean(index-1, index, index+1)

        avoid_regions: list of list (default None)
            Defines wavelength regions that are avoided in the selection when space
            is an integer. The sub_list should have two elements defining the region
            to avoid in wavelength values
            i. e.: [[380,450],[520,530] traces with wavelength values between 380-450
                   and 520-530 will not be selected

        Returns
        ----------
        Modifies the attributes selected_traces and selected_wavelength.
        """
        if points == 'all':
            self.selected_traces, self.selected_wavelength = copy(
                self.data), copy(self.wavelength)
        else:
            self.selected_traces, self.selected_wavelength = select_traces(
                self.data, self.wavelength, points, average, avoid_regions)
        self._SVD_fit = False
