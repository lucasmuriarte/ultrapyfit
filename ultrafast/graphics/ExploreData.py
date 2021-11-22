# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:18:25 2020
@author: lucas
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from ultrafast.utils.divers import select_traces, FiguresFormating, \
    TimeUnitFormater
from ultrafast.utils.Preprocessing import ExperimentException
from ultrafast.graphics.MaptplotLibCursor import SnaptoCursor
import pandas as pd
from ultrafast.graphics.PlotSVD import PlotSVD
from copy import copy
import matplotlib.cm as cm
from ultrafast.graphics.styles.set_styles import *
from ultrafast.graphics.styles.plot_base_functions import *


class ExploreData(PlotSVD):
    """
    Class to explore a time resolved data set and easily create figures already
    formatted. The class inherits PlotSVD therefore all methods for plotting and
    exploring the SVD space are available.
    Attributes
    ----------
    x: 1darray
        x-vector, normally time vector
    data: 2darray
        array containing the data, the number of rows should be equal to
        the len(x)
    wavelength: 1darray
        wavelength vector
    selected_traces: 2darray
        sub dataset of data
    selected_wavelength: 1darray
        sub dataset of wavelength
    time_unit: str (default ps)
        Contains the strings of the time units to format axis labels and legends
        automatically:
        time_unit str of the unit. >>> e.g.: 'ps' for picosecond
        Can be passes as kwargs when instantiating the object
    wavelength_unit: str (default nm)
        Contains the strings of the wavelength units to format axis labels and
        legends automatically:
        wavelength_unit str of the unit: >>> e.g.: 'nm' for nanometers
        Can be passes as kwargs when instantiating the object
    cmap: str
        Contains the name of a matplotlib color map
    """

    def __init__(self,
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
            array containing the data, the number of rows should be equal
            to the len(x)
        wavelength: 1darray
            wavelength vector
        selected_traces: 2darray (default None)
            sub dataset of data
        selected_wavelength: 1darray (default None)
            sub dataset of wavelength
        cmap: str
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
            self.selected_wavelength = wavelength
        else:
            self.selected_traces = selected_traces
            self.selected_wavelength = selected_wavelength
        self._units = units
        self._SVD_fit = False
        self._unit_formater = TimeUnitFormater(self._units['time_unit'])
        self._color_map = cmap
        super().__init__(self.x, self.data, self.wavelength,
                         self.selected_traces, self.selected_wavelength)

    def get_color_map(self):
        return self._color_map

    def set_color_map(self, val: str):
        if val in cm.__dict__.keys():
            self._color_map = val
        else:
            msg = 'Not a valid color map, check matplotlib for valid names'
            raise ExperimentException(msg)

    @property
    def time_unit(self):
        return f'{self._unit_formater._multiplicator.name}s'

    @property
    def wavelength_unit(self):
        return self._units['wavelength_unit']

    @time_unit.setter
    def time_unit(self, val: str):
        try:
            val = val.lower()
            self._unit_formater.multiplicator = val
            self._units['time_unit'] = self.time_unit
        except Exception:
            msg = 'An unknown time unit cannot be set'
            raise ExperimentException(msg)

    @wavelength_unit.setter
    def wavelength_unit(self, val: str):
        val = val.lower()
        if 'nanom' in val or 'wavelen' in val:
            val = 'nm'
        if 'centim' in val or 'wavenum' in val or 'cm' in val:
            val = 'cm-1'
        self._units['wavelength_unit'] = val

    @use_style
    def plot_traces(self, traces='select', style='lmu_trac', legend='auto'):
        """
        Plots either the selected traces or 9-10 trace equally space in the
        wavelength range. If less than 10 (included) traces are plotted a
        legend will be display.

        Parameters
        ----------
        traces: auto, select or a list  (default select)
            1--> If auto between 9 and 10 traces equally spaced in the
                wavelength range will be plotted.
            2--> If select the selected traces will be plotted
            3--> If a list containing the traces wave values

        style: style valid name (default 'lmu_res')
            defines the style to format the output figure, it can be any defined
            matplotlib style, any ultrafast style (utf) or any user defined
            style that follows matplotlib or utf styles and is saved in the
            correct folder. Check styles for more information.

        legend: "auto", True or False, (default "auto")
            If auto, a legend is display if there are less than 10 traces in the
            Figure plotted.
            If True or False, a legend will be displayed or not independently of
            the number of traces in the Figure.

        Returns
        ----------
        Figure and axes matplotlib objects
        """
        data, values = self._get_traces_to_plot(traces)
        fig, ax = plt.subplots(1)
        alpha = 0.60
        for i in range(len(values)):
            ax.plot(self.x, data[:, i], marker='o', alpha=alpha, ms=4, ls='')
        if legend == 'auto':
            if len(values) <= 10 or traces == 'auto':
                legenda = self._traces_legend(traces, values)
                ax.legend(legenda, loc='best', ncol=2)
        elif legend:
            legenda = self._traces_legend(traces, values)
            ax.legend(legenda, loc='best', ncol=2)
        FiguresFormating.axis_labels(ax, f'Time ({self._units["time_unit"]})',
                                     '$\Delta$A')
        return fig, ax

    @use_style
    def plot_spectra(self,
                     times='all',
                     rango=None,
                     average=0,
                     cover_range=None,
                     from_max_to_min=True,
                     legend=True,
                     legend_decimal=2,
                     ncol=1,
                     cmap=None,
                     style='lmu_spec',
                     include_rango_max=True):
        """
        Function to plot spectra

        Parameters
        ----------
        times: list or "all" or "auto" (default "all")
            "all": all spectra will be plotted if they are less than 250
                   spectra (if there are more than 250 the output is the same
                   as "auto")
            "auto": 8 spectra equally spaced at the wavelength where the data
                   has maximum amplitude will be displayed
            list: (option 1)
                list containing a number of time points, then the closest points
                in x vector to those in the list will be plotted
            list: (Option 2) --> modifies auto plotting
                a list of either length 2 or 3 elements where the structure
                should follow this sequence:
                mode = "auto"
                number_of_spectra = (optional; int)
                wavelength_to_select_spectra = (optional; int)
                times=["auto", number_of_spectra,  wavelength_to_select_spectra]
                with three possible options:
                1 -->if only ["auto"] 8 spectra will be plotted equally spaced
                     at the maximum for all wavelengths same as "auto"
                     as string
                2 -->if ["auto", n_number_spec] n_number spectra will be plotted
                     equally spaced at the maximum for all wavelengths
                3 -->if ["auto", n_number_spec, wavelength] n_number spectra
                     will be plotted equally spaced at the selected wavelength
                e.g.1: ["auto", 10, 350] >>> 10 spectra equally spaced at
                                            wavelength 350
                e.g.2: ["auto", 10] >>> 10 spectra equally spaced at data
                                        wavelength were signal is maximum
                                        amplitude (absolute value)
                 e.g.3: ["auto", 6, 480] >>> 6 spectra equally spaced at w
                                             avelength 480

        rango: List of length 2 or None (default None)
            Defines a time range where to auto select spectra

        average: int (default 0)
            Number of points to average the spectra. Is recommended to be zero
            except for special cases

        cover_range: List of length 2 or None (default None)
            Defines a range in wavelength that will be cover in white. This can
            be use to cut the excitation wavelength range

        from_max_to_min: bool (default True)
            In case the rango is None. Generally the data sets have a time point
            where the transient signal is maximum. If from_max_to_min is True
            the range will be given from the time point where the data amplitude
            is maximum to the end. If False from the initial time to the maximum
            in amplitude of the data.

        legend: True, False or bar
            If True a legend will be display.
            If False no legend will be display.
            If bar a color-bar with the times is add to the side.

        legend_decimal: int (default 2)
            Only applicable if legend=True
            number of decimal values in the legend names

        ncol: int (default 1)
            Number of legend columns
        cmap: str or None (default None)

            Name of matplotlib color map if None the attribute color_map will
            be use

        style: style valid name (default 'lmu_res')
            defines the style to format the output figure, it can be any defined
            matplotlib style, any ultrafast style (utf) or any user defined
            style that follows matplotlib or utf styles and is saved in the
            correct folder. Check styles for more information.
        include_rango_max: bool (default True)

            If True, spectra are auto-plotted in a given range the last spectrum
            plotted will be the closest to the range limit

        Returns
        ----------
        Figure and axes matplotlib objects
        """
        data = self.data
        wavelength = self._get_wavelength()
        times, legend, average = self._verify_plot_spectra(times, data,
                                                           legend, average)
        times = self._time_to_real_times(times, rango, include_rango_max,
                                         from_max_to_min)
        legenda = [self._unit_formater.value_formated(i, legend_decimal)
                   for i in times]
        colors = self._get_color(times, cmap)
        fig, ax = plt.subplots(1)
        tiempo = pd.Series(self.x)
        for i in range(len(times)):
            index = (tiempo - times[i]).abs().sort_values().index[0]
            if average != 0:
                trace = np.mean(data[index - average:index + average, :],
                                axis=0)
            else:
                trace = data[index, :]
            ax.plot(wavelength, trace, c=colors[i], label=legenda[i])
        self._format_spectra_figure(ax, cover_range)
        self._legend_spectra_figure(legend, ncol, cmap, times)
        return fig, ax

    @use_style
    def plot_3D(self, cmap=None, figsize=(12, 8)):
        """
        Plot the data in 3D

        Parameters
        ----------
        cmap: str or None
            name of matplotlib color map if None the attribute color_map will
             be use

        figsize: tuple
            Size of the figure similar to matplotlib figsize

        Returns
        ----------
        Figure and axes matplotlib objects
        """
        cmap = self._get_cmap(cmap)
        x = self.x
        z = self.data.transpose()
        y = self.wavelength
        if self._units["wavelength_unit"] == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self._units["wavelength_unit"]})'
        x, y = np.meshgrid(x, y)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, cmap=cmap,
                               linewidth=0, antialiased=False)
        ax.set_zlim(np.min(z), np.max(z))
        ax.set_xlabel(f'Time ({self._units["time_unit"]})')
        ax.set_ylabel(xlabel)
        ax.set_zlabel('$\Delta$A')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig, ax

    def select_traces(self, points=10, average=1, avoid_regions=None):
        """
        Method to select traces from the data attribute and defines a subset
        of traces in the selected_traces attribute.

        Parameters
        ----------
        points: int or list or "auto" (default 10)
            If type(space) =int: a series of traces separated by the value
            indicated will be selected.
            If type(space) = list: the traces in the list will be selected.
            If space = auto, the number of returned traces is 10 and equally
            spaced along the wavelength vector and points is set to 0
        average: int (default 1)
            Binning points surrounding the selected wavelengths.
            e. g.: if point is 1 trace = mean(index-1, index, index+1)
        avoid_regions: list of list (default None)
            Defines wavelength regions that are avoided in the selection when
            space is an integer. The sub_list should have two elements defining
            the region to avoid in wavelength values
            i. e.: [[380,450],[520,530] traces with wavelength values
                    between 380-450 and 520-530 will not be selected
        Returns
        ----------
        Modifies the attributes selected_traces and selected_wavelength.
        """
        if hasattr(points, '__iter__'):
            if len(points) == len(self.wavelength):
                points = 'all'
        super().select_traces(points, average, avoid_regions)


    def _verify_plot_spectra(self, times, data, legend, average):
        """
        check plot spectra conditions
        """
        if times == 'all' or times == 'auto' or type(times) == list:
            if data.shape[0] > 250 and times == 'all':
                times = 'auto'
                msg = 'More than 250 spectra cannot be plotted or your ' \
                      'computer risk of running out of memory'
                print(msg)
            elif times == 'all':
                legend = False
                average = 0
            return times, legend, average
        else:
            statement = 'times should be either: 1-->"all" \n 2-->"auto" \n \
                        3-->list with selected point to plot \n \
                        4--> list this form ["auto", Nº spectra, wavelength]'
            raise ExperimentException(statement)

    def _get_traces_to_plot(self, traces):
        if traces == 'auto':
            number_traces = len(self.wavelength) // 10
            values = [i for i in range(len(self.wavelength))[::number_traces]]
            data = self.data
        elif traces == 'select':
            data = self.selected_traces

            # if self.selected_wavelength is not None and self._SVD_fit is False:
            #    values = np.where(np.in1d(self.wavelength,
            #                              self.selected_wavelength))[0]
                # print(values[0])
            # else:
            values = [i for i in range(data.shape[1])]
        elif type(traces) == list:
            values = [np.argmin(abs(self.wavelength - i)) for i in traces]
            data = self.data
        return data, values

    def _traces_legend(self, traces, values):
        """
        return the formatted legend for trace pltting
        """
        if self._SVD_fit and traces != 'auto':
            legenda = ['left SV %i' % i
                       for i in range(1, self.data.shape[1] + 1)]
        elif self.selected_wavelength is not None:
            if self.wavelength_unit == 'cm-1':
                val = 'cm$^{-1}$'
            else:
                val = self.wavelength_unit
            legenda = [f'{round(i)} {val}' for i in self.wavelength[values]]
        else:
            legenda = [f'curve {i}' for i in range(self.data.shape[1])]
        return legenda

    def select_traces_graph(self, points=-1, average=0):
        """
        Function to select traces graphically

        Parameters
        ----------
        points: int (default -1)
            Defines the number of traces that can be selected. If -1 an
            infinitive number of traces can be selected
        """
        fig, ax = self.plot_spectra()
        cursor = SnaptoCursor(ax, self.wavelength, self.wavelength * 0.0,
                              points)
        plt.connect('axes_enter_event', cursor.onEnterAxes)
        plt.connect('axes_leave_event', cursor.onLeaveAxes)
        plt.connect('motion_notify_event', cursor.mouseMove)
        plt.connect('button_press_event', cursor.onClick)
        fig.canvas.mpl_connect('close_event',
                               lambda event: self.select_traces(cursor.datax,
                                                                average))
        return fig, cursor

    def _legend_spectra_figure(self, legend, ncol, cmap, times):
        if legend == "bar":
            cnorm = Normalize(vmin=times[0], vmax=times[-1])
            cpickmap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            cpickmap.set_array([])
            plt.colorbar(cpickmap).set_label(
                label='Time (' + self._units["time_unit"] + ')', size=15)
        elif legend:
            leg = plt.legend(loc='best', ncol=ncol)
            leg.set_zorder(np.inf)
        else:
            pass

    def _get_wavelength(self):
        if self.wavelength is not None:
            wavelength = self.wavelength
        else:
            wavelength = np.array([i for i in range(len(self.data[1]))])
        return wavelength

    def _get_cmap(self, cmap):
        if cmap is None:
            cmap = self._color_map
        return cmap

    def _get_color(self, times, cmap=None):
        """
        return colors to use in a plot
        """
        cmap = self._get_cmap(cmap)
        a = np.linspace(0, 1, len(times))
        c = plt.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = c.to_rgba(a, norm=False)
        return colors

    def _time_to_real_times(self, times, rango, include_max, from_max_to_min):
        """
        Method to read the passed argument times and pass this argument
        to _getAutoPoints
        """
        if times[0] == 'auto' or times == 'auto':
            if times == 'auto':
                times = ['auto']
            if len(times) == 1:
                times = self._get_auto_points(rango=rango,
                                              include_rango_max=include_max,
                                              decrease=from_max_to_min)
            elif len(times) == 2:
                times = self._get_auto_points(spectra=times[1],
                                              rango=rango,
                                              include_rango_max=include_max,
                                              decrease=from_max_to_min)
            elif len(times) == 3:
                times = self._get_auto_points(times[1], times[2],
                                              rango=rango,
                                              include_rango_max=include_max,
                                              decrease=from_max_to_min)
            else:
                msg = 'for auto ploting times should follow this structure:\n\
                      mode = "auto"\n\
                      Nºspectra = (optional; int)\n\
                      wavelength_to_select_spectra = (optional; int)\n\
                      times=["auto", Nºspectra, wavelength_to_select_spectra]\n\
                      e.g: times=["auto", 6, 480]\n\
                      >>> 6 spectra equally spaced at wavelength 480'
                raise ExperimentException(msg)

        elif times == 'all':
            times = self.x
        else:
            times = self.x[[np.argmin(abs(self.x - i)) for i in times]]
        times = sorted(list(set(times)))
        return times

    def _format_spectra_figure(self, ax, cover_range):
        # FiguresFormating.format_figure(ax, data, wavelength,
        #                                size=size, x_tight=True,
        #                                set_ylim=False)
        FiguresFormating.axis_labels(ax, self._get_wave_label(), '$\Delta$A')
        if cover_range is not None:
            FiguresFormating.cover_excitation(ax, cover_range, self.wavelength)

    def _get_wave_label(self):
        """
        Returns a formatted string from the units attribute
        """
        if self.wavelength is None:
            xlabel = 'pixel'
        elif self._units['wavelength_unit'] == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self._units["wavelength_unit"]})'
        return xlabel

    def _get_auto_points(self, spectra=8, wave=None, rango=None,
                         include_rango_max=True, decrease=True):
        """
        Returns list of time points (spectra) equally spaced spectra in
        amplitude at the wavelength (wave)in the time ranges (rango).
        If include_rango_max the last time point corresponds to the high limit
        value of rango. In case rango is None and there is a increase and
        decrease sequence in the data set (e.g.: Signal formation follow by
        decrease of the transient signal) the selection can be done with the
        parameter decrease, In case True the range will be given from the tome
        where the data amplitude is maximum to the end. If False from the
        initial time to the maximum in amplitude of the data.
        """
        data = self.data
        wavelength = self.wavelength
        x = self.x
        if rango is not None:
            if type(rango) is list and len(rango) != 2:
                msg = 'rango should be None or a list containing the minimum ' \
                      'and maximum value of the range to select points' \
                      'e.g.: rango = [5,500]'
                raise ExperimentException(msg)
            rango = sorted(rango)
            mini = np.argmin([abs(i - rango[0]) for i in x])
            maxi = np.argmin([abs(i - rango[1]) for i in x])
            x = x[mini:maxi + 1]
            data = data[mini:maxi + 1, :]
        if wave is None:
            idx = np.unravel_index(np.argmax(abs(data), axis=None), data.shape)
            # get points from the maximum of the trace
            # (at the maximum wavelenght) to the end
            if decrease and rango is None:
                equally_spaced = np.linspace(np.min(data[idx[0]:, idx[1]]),
                                             np.max(data[idx[0]:, idx[1]]),
                                             spectra)
                point = [idx[0] + np.argmin(abs(data[idx[0]:, idx[1]] - i))
                         for i in equally_spaced]

            else:
                # get points from the minimum of the trace
                # (at the maximum wavelength) to the maximum
                equally_spaced = np.linspace(np.min(data[:, idx[1]]),
                                             np.max(data[:, idx[1]]),
                                             spectra)
                point = [0 + np.argmin(abs(data[:, idx[1]] - i))
                         for i in equally_spaced]

                if 0 not in point:
                    point[np.argmin(point)] = 0
            if rango is not None and include_rango_max:
                point[np.argmax(point)] = -1
            print(wavelength[idx[1]])
            return np.sort(np.array(x)[point])
        else:
            if wavelength is not None:
                wave_idx = np.argmin(abs(np.array(wavelength) - wave))
                idx = np.argmax(abs(data[:, wave_idx]))
                if decrease and rango is None:
                    equally_spaced = np.linspace(np.min(data[idx:, wave_idx]),
                                                 np.max(data[idx:, wave_idx]),
                                                 spectra)
                    point = [idx + np.argmin(abs(data[idx:, wave_idx] - i))
                             for i in equally_spaced]
                else:
                    equally_spaced = np.linspace(np.min(data[:, wave_idx]),
                                                 np.max(data[:, wave_idx]),
                                                 spectra)
                    point = [0 + np.argmin(abs(data[:, wave_idx] - i))
                             for i in equally_spaced]
                    if 0 not in point:
                        point[np.argmin(point)] = 0
                if rango is not None and include_rango_max:
                    point[np.argmax(point)] = -1
                print(wavelength[wave_idx])
                return np.sort(np.array(x)[point])
            else:
                msg = 'Wavelength is not defined'
                raise ExperimentException(msg)