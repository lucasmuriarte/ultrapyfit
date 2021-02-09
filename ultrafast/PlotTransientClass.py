# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:18:25 2020

@author: lucas
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from ultrafast.tools import select_traces, FiguresFormating, TimeUnitFormater, TimeMultiplicator
from ultrafast.PreprocessingClass import ExperimentException
from ultrafast.MaptplotLibCursor import SnaptoCursor
import pandas as pd
from ultrafast.PlotSVDClass import PlotSVD
from copy import copy


class ExploreData(PlotSVD):
    """
    Class to explore a time resolved data set and easily create figures already formatted.
    The class inherits PlotSVD therefore all methods for plotting and exploring the SVD
    space are available.

    Attributes
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2darray
        array containing the data, the number of rows should be equal to the len(x)

    wavelength: 1darray
        wavelength vector

    params: lmfit parameters object
        object containing the initial parameters values used to build an exponential model.
        These parameters are iteratively optimize to reduce the residual matrix formed by
        data-model (error matrix) using Levenberg-Marquardt algorithm.

    selected_traces: 2darray
        sub dataset of data

    selected_wavelength: 1darray
        sub dataset of wavelength

    units: dictionary
        This dictionary contains the strings of the units to format axis labels and legends automatically:
        "time_unit": str of the unit. >>> e.g.: 'ps' for picosecond
        "wavelength_unit": wavelength unit
        The dictionary keys can be passes as kwargs when instantiating the object

    color_map: str
        Contains the name of a matplotlib color map
    """

    def __init__(
            self,
            x,
            data,
            wavelength,
            selected_traces=None,
            selected_wavelength=None,
            cmap='viridis',
            **kwargs):
        """
        Constructor

        Parameters
        ----------
        x: 1darray
            x-vector, normally time vector

        data: 2darray
            array containing the data, the number of rows should be equal to the len(x)

        wavelength: 1darray
            wavelength vector

        selected_traces: 2darray (default None)
            sub dataset of data

        selected_wavelength: 1darray (default None)
            sub dataset of wavelength

        color_map: str
            defines the general color map

        kwargs:
            units attribute keys
        """
        units = dict({'time_unit': 'ps', 'wavelength_unit': 'nm'}, **kwargs)
        self.data = data
        self.x = x
        self.wavelength = wavelength
        if selected_traces is None:
            self.selected_traces = data
            self.selected_wavelength = None
        else:
            self.selected_traces = selected_traces
            self.selected_wavelength = selected_wavelength
        self.units = units
        self._SVD_fit = False
        self._unit_formater = TimeUnitFormater(self.units['time_unit'])
        self._cursor = None
        self._fig_select = None
        self.color_map = cmap
        super().__init__(
            self.x,
            self.data,
            self.wavelength,
            self.selected_traces,
            self.selected_wavelength)
        # PlotSVD.__init__(self.x, self.data, self.wavelength, self.selected_traces, self.selected_wavelength)

    def plot_traces(self, auto=False, size=14):
        """
        Plots either the selected traces or 9-10 trace equally space in the wavelength range.
        If less than 10 (included) traces are plotted a legend will be display.

        Parameters
        ----------
        auto: bool (default False)
            If True between 9 and 10 traces equally spaced in the wavelength range will be plotted.
            If False the selected traces will be plotted

        size: int (default 14)
            size of the figure text labels including tick labels axis labels and legend

        Returns
        ----------
        Figure and axes matplotlib objects
        """
        if auto:
            values = [i for i in range(len(self.wavelength))[
                ::len(self.wavelength) // 10]]
        else:
            if self.selected_wavelength is not None:
                values = np.where(
                    np.in1d(
                        self.wavelength,
                        self.selected_wavelength))
            else:
                values = [i for i in range(self.data.shape[1])]
        if len(values) <= 10 or auto:
            if self._SVD_fit:
                legenda = [
                    'left SV %i' %
                    i for i in range(
                        1,
                        self.data.shape[1] +
                        1)]
            elif self.selected_wavelength is not None:
                legenda = [
                    f'{round(i)} {self.units["wavelength_unit"]}' for i in self.wavelength[values]]
            else:
                legenda = [f'curve {i}' for i in range(self.data.shape[1])]
        fig, ax = plt.subplots(1, figsize=(11, 6))
        alpha = 0.60
        for i in values:
            ax.plot(self.x, self.data[:, i],
                    marker='o', alpha=alpha, ms=4, ls='')
        if self.data.shape[1] <= 10 or auto:
            ax.legend(legenda, loc='best', ncol=2)
        FiguresFormating.format_figure(ax, self.data, self.x, size=size)
        FiguresFormating.axis_labels(
            ax,
            f'Time ({self.units["time_unit"]})',
            '$\\Delta$A',
            size=size)
        return fig, ax

    def plot_spectra(
            self,
            times='all',
            rango=None,
            n_points=0,
            cover_range=None,
            from_max_to_min=True,
            legend=True,
            legend_decimal=2,
            ncol=1,
            cmap=None,
            size=14,
            include_rango_max=True):
        """
        Function to plot spectra

        Parameters
        ----------
        times: list or "all" or "auto" (default "all")
            "all": all spectra will be plotted if they are less than 250 spectra (if there are more than 250 the output
                    is the same as auto

            "auto": 8 spectra equally spaced at the wavelength where the data has maximum amplitude will be displayed

            list: (option 1)
                list containing a number of time points, then the closest points in x vector to those in the list
                will be plotted

            list: (Option 2)
                a list of either length 2 or 3 elements where the structure should follow this sequence:
                times=["auto", number_of_spectra(optional; int),  wavelength_to_select_spectra(optional; int)],

                with three possible options:
                1 -->if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths
                    same as "auto"  as string
                2 -->if ["auto", n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all
                        wavelengths
                3 -->if ["auto", n_number_spec, wavelength] n_number spectra will be plotted equally spaced at the
                        selected wavelength
                e.g.1: ["auto", 10, 350] >>> 10 spectra equally spaced at wavelength 350
                e.g.2: ["auto", 10] >>> 10 spectra equally spaced at data wavelength were signal is
                                        maximum amplitude (absolute value)
                 e.g.3: ["auto", 6, 480] >>> 6 spectra equally spaced at wavelength 480

        rango: List of length 2 or None (default None)
            Defines a time range where to auto select spectra

        n_points: int (default 0)
            Number of points to average the spectra. Is recommended to be zero except for special cases

        cover_range: List of length 2 or None (default None)
            Defines a range in wavelength that will be cover in white. This can be use to cut the excitation
            wavelength range

        from_max_to_min: bool (default True)
            In case the rango is None. Generally the data sets have a time point where the transient signal is maximum.
            If from_max_to_min is True the range will be given from the time point where the data amplitude is maximum
            to the end. If False from the initial time to the maximum in amplitude of the data.

        legend: bool (default True)
            If True a legend will be display, If False color-bar with the times will be display

        legend_decimal: int (default 2)
            number of decimal values in the legend names

        ncol: int (default 1)
            Number of legend columns

        cmap: str or None (default None)
            Name of matplotlib color map if None the attribute color_map will be use

        size: int (default 14)
            size of the figure text labels including tick labels axis labels and legend

        include_rango_max: bool (default True)
            If True, spectra are auto-plotted in a given range the last spectrum plotted will be the
            closest to the range limit

        Returns
        ----------
        Figure and axes matplotlib objects
        """
        if times == 'all' or times == 'auto' or isinstance(times, list):
            if cmap is None:
                cmap = self.color_map
            data = self.data
            wavelength = self.wavelength if self.wavelength is not None else np.array(
                [i for i in range(len(data[1]))])
            if data.shape[0] > 250 and times == 'all':
                times = 'auto'
                print(
                    'More than 250 spectra cannot be plotted or your computer risk of running out of memory')
            elif times == 'all':
                legend = False
                n_points = 0
            times = self._time_to_real_times(
                times, rango, include_rango_max, from_max_to_min)
            legenda = [
                self._unit_formater.value_formated(
                    i, legend_decimal) for i in times]
            a = np.linspace(0, 1, len(times))
            c = plt.cm.ScalarMappable(norm=None, cmap=cmap)
            colors = c.to_rgba(a, norm=False)
            fig, ax = plt.subplots(1, figsize=(11, 6))
            tiempo = pd.Series(self.x)
            for i in range(len(times)):
                index = (tiempo - times[i]).abs().sort_values().index[0]
                if n_points != 0:
                    ax.plot(wavelength, np.mean(
                        data[index - n_points:index + n_points, :], axis=0), c=colors[i], label=legenda[i])
                else:
                    ax.plot(wavelength, data[index, :],
                            c=colors[i], label=legenda[i])
            FiguresFormating.format_figure(
                ax, data, wavelength, size=size, x_tight=True, set_ylim=False)
            FiguresFormating.axis_labels(
                ax, self._get_wave_label(), '$\\Delta$A', size=size)

            if legend:
                leg = plt.legend(loc='best', ncol=ncol, prop={'size': size})
                leg.set_zorder(np.inf)
            else:
                cnorm = Normalize(vmin=times[0], vmax=times[-1])
                cpickmap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
                cpickmap.set_array([])
                plt.colorbar(cpickmap).set_label(
                    label='Time (' + self.units["time_unit"] + ')', size=15)
            if cover_range is not None:
                FiguresFormating.cover_excitation(
                    ax, cover_range, self.wavelength)
            return fig, ax
        else:
            statement = 'times should be either "all" or "auto" \n \
            or a list with selected point to plot \n \
            or a list this form ["auto", number of spectra(optional; int), wavelength to select spectra(optional; int)] \n \
            if times is a list and the first element is "auto" then spectra will be auto plotted \n \
            times should have this form:\n\
            times=["auto", number_of_spectra(optional; int),  wavelength_to_select_spectra(optional; int)],\n \
            with three possible options:\n\
            1 -->if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths\n \
            2 -->if ["auto",n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all wavelengths\n \
            3 -->if ["auto",n_number_spec,wavelength] n_number spectra will be plotted equally spaced at the selected ' \
                        'wavelength'
            raise ExperimentException(statement)

    def plot3D(self, cmap=None):
        """
        Plot the data in 3D

        Parameters
        ----------
        cmap: str or None
            name of matplotlib color map if None the attribute color_map will be use

        Returns
        ----------
        Figure and axes matplotlib objects
        """
        if cmap is None:
            cmap = self.color_map
        X = self.x
        Z = self.data.transpose()
        Y = self.wavelength
        if self.units["wavelength_unit"] == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self.units["wavelength_unit"]})'
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                               linewidth=0, antialiased=False)
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.set_xlabel(f'Time ({self.units["time_unit"]})')
        ax.set_ylabel(xlabel)
        ax.set_zlabel('$\\Delta$A')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig, ax

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

    def select_traces_graph(self, points=-1, average=0):
        """
        Function to select traces graphically

        Parameters
        ----------
        points: int (default -1)
            Defines the number of traces that can be selected. If -1 an infinitive number of traces can be selected
        """
        self._fig_select, ax = self.plot_spectra()
        self._cursor = SnaptoCursor(
            ax, self.wavelength, self.wavelength * 0.0, points)
        plt.connect('axes_enter_event', self._cursor.onEnterAxes)
        plt.connect('axes_leave_event', self._cursor.onLeaveAxes)
        plt.connect('motion_notify_event', self._cursor.mouseMove)
        plt.connect('button_press_event', self._cursor.onClick)
        self._fig_select.canvas.mpl_connect(
            'close_event', self.select_traces(
                self._cursor.datax, average))

    def _time_to_real_times(self, times, rango, include_max, from_max_to_min):
        """
        Method to read the passed argument times and pass this argument to _getAutoPoints
        """
        if times[0] == 'auto':
            if len(times) == 1:
                times = self._get_auto_points(
                    rango=rango,
                    include_rango_max=include_max,
                    decrease=from_max_to_min)
            elif len(times) == 2:
                times = self._get_auto_points(
                    spectra=times[1],
                    rango=rango,
                    include_rango_max=include_max,
                    decrease=from_max_to_min)
            elif len(times) == 3:
                times = self._get_auto_points(
                    times[1],
                    times[2],
                    rango=rango,
                    include_rango_max=include_max,
                    decrease=from_max_to_min)
            else:
                print(
                    'if first element is "auto" then spectra will be auto plotted \n \
                      then the list can be only   ["auto"] or:\n\
                      ["auto", number of spectra(optional; int),  wavelength to select spectra(optional; int)],\n \
                      if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all '
                    'wavelengths\n \
                      if ["auto",15,wavelength] 15 spectra will be plotted equally spaced at the selected wavelength')

        elif times == 'all':
            times = self.x
        elif times == 'auto':
            times = self._get_auto_points(
                rango=rango,
                include_rango_max=include_max,
                decrease=from_max_to_min)
        else:
            times = self.x[[np.argmin(abs(self.x - i)) for i in times]]
        times = sorted(list(set(times)))
        return times

    def _get_wave_label(self):
        """
        Returns a formatted string from the units attribute
        """
        if self.wavelength is None:
            xlabel = 'pixel'
        elif self.units['wavelength_unit'] == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self.units["wavelength_unit"]})'
        return xlabel

    def _get_auto_points(
            self,
            spectra=8,
            wave=None,
            rango=None,
            include_rango_max=True,
            decrease=True):
        """
        Returns list of time points (spectra) equally spaced spectra in amplitude at the wavelength (wave)
        in the time ranges (rango). If include_rango_max the last time point corresponds to the high limit
        value of rango. In case rango is None and there is a increase and decrease sequence in the data set
        (e.g.: Signal formation follow by decrease of the transient signal) the selection can be done with
        the parameter decrease, In case True the range will be given from the tome where the data amplitude
        is maximum to the end. If False from the initial time to the maximum in amplitude of the data.
        """
        data = self.data
        wavelength = self.wavelength
        x = self.x
        if rango is not None:
            if isinstance(rango, list) and len(rango) != 2:
                statement = 'rango should be None or a list containing the minimum and maximum value of the range ' \
                            'to select points'
                raise ExperimentException(statement)
            rango = sorted(rango)
            mini = np.argmin([abs(i - rango[0]) for i in x])
            maxi = np.argmin([abs(i - rango[1]) for i in x])
            x = x[mini:maxi + 1]
            data = data[mini:maxi + 1, :]
        if wave is None:
            idx = np.unravel_index(np.argmax(abs(data), axis=None), data.shape)
            # get points from the maximum of the trace (at the maximum
            # wavelenght) to the end
            if decrease and rango is None:
                point = [idx[0] + np.argmin(abs(data[idx[0]:, idx[1]] - i)) for i in np.linspace(
                    np.min(data[idx[0]:, idx[1]]), np.max(data[idx[0]:, idx[1]]), spectra)]
            else:
                # get points from the minimum of the trace (at the maximum
                # wavelength) to the maximum
                point = [0 + np.argmin(abs(data[:, idx[1]] - i)) for i in np.linspace(
                    np.min(data[:, idx[1]]), np.max(data[:, idx[1]]), spectra)]
                if 0 not in point:
                    point[np.argmin(point)] = 0
            if rango is not None and include_rango_max:
                point[np.argmax(point)] = -1
#           print (wavelength[idx[1]])
            return np.sort(np.array(x)[point])
        else:
            if wavelength is not None:
                wave_idx = np.argmin(abs(np.array(wavelength) - wave))
                idx = np.argmax(abs(data[:, wave_idx]))
                if decrease and rango is None:
                    point = [idx + np.argmin(abs(data[idx:, wave_idx] - i)) for i in np.linspace(
                        np.min(data[idx:, wave_idx]), np.max(data[idx:, wave_idx]), spectra)]
                else:
                    point = [0 + np.argmin(abs(data[:, wave_idx] - i)) for i in np.linspace(
                        np.min(data[:, wave_idx]), np.max(data[:, wave_idx]), spectra)]
                    if 0 not in point:
                        point[np.argmin(point)] = 0
                if rango is not None and include_rango_max:
                    point[np.argmax(point)] = -1
                print(wavelength[wave_idx])
                return np.sort(np.array(x)[point])
            else:
                print('Wavelength is not defined')
