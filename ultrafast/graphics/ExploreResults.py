# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:35:41 2020

@author: lucas
"""
from ultrafast.utils.divers import FiguresFormating, solve_kmatrix, TimeUnitFormater
from ultrafast.utils.Preprocessing import ExperimentException
import matplotlib.pyplot as plt
import numpy as np
from ultrafast.fit.ModelCreator import ModelCreator
import scipy.integrate as integral
from matplotlib.widgets import Slider


class ExploreResults(FiguresFormating):
    def __init__(self, fits, **kwargs):
        units = dict({'time_unit': 'ps', 'wavelength_unit': 'nm'}, **kwargs)
        if type(fits) == dict:
            self._fits = fits
        else:
            self._fits = {1: fits}
        self._units = units
        self._unit_formater = TimeUnitFormater(self._units['time_unit'])
        self._l = None
        self._ll = None
        self._lll = None
        self._fig = None
        self._residues = None
        self._data_fit = None
        self._x_fit = None
        self._fittes = None
        self._sspec = None
        self._x_verivefit = None
        self._residues = None
        self._wavelength_fit = None
        self._title = None

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
            self._units['time_unit'] = val
            self._unit_formater.multiplicator = val
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

    def results(self, fit_number=None, verify_svd_fit=False):
        """
        Returns a data set created from the best parameters values.

        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered

        verify_svd_fit: bool (default  False)
            If True it will return the single fit perform to every trace of the spectra after an svd fit
            If false and the fit is an SVD, the values return are the fit to the svd
            If is not an SVD fit this parameter is not applicable
        """
        x, data, wavelength, result_params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space,  = \
            self._get_values(fit_number=fit_number,
                             verify_svd_fit=verify_svd_fit)
        model = ModelCreator(exp_no, x, tau_inf)
        ndata, nx = data.shape
        if type_fit == 'Exponential':
            if deconv:
                curve_resultados = data * 0.0
                for i in range(nx):
                    curve_resultados[:, i] = model.expNGaussDataset(result_params, i)
            else:
                t0 = result_params['t0_1'].value
                index = np.argmin([abs(i - t0) for i in x])
                curve_resultados = 0.0 * data[index:, :]
                for i in range(nx):
                    curve_resultados[:, i] = model.expNDataset(result_params, i)
        else:
            coeffs, eigs, eigenmatrix = solve_kmatrix(exp_no, result_params)
            curve_resultados = data * 0.0
            for i in range(nx):
                curve_resultados[:, i] = model.expNGaussDatasetTM(result_params,
                                                                  i,
                                                                  [coeffs,
                                                                   eigs,
                                                                   eigenmatrix])
        return curve_resultados

    def plot_fit(self, fit_number=None, selection=None, plot_residues=True, size=14,):
        """
        Function that generates a figure with the results of the fit stored in the all_fit attributes
        If less than 10 traces are fitted or selected a legend will be displayed
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered
        
        selection: list or None (default None)    
            If None all the traces fitted will be plotted, if not only those selected in the lis
        
        plot_residues: Bool (default True)
            If True the Figure returned will contain two axes a top one with the residues, and the bottom one
            with the fit and data
            
        size: int (default 14)
            size of the figure text labels including tick labels axis labels and legend
        
        Returns
        ----------
        Figure and axes matplotlib objects
        """
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)

        if wavelength is None:
            wavelength = np.array([i for i in range(len(data[1]))])
        if svd_fit:
            selection = None
        if selection is None:
            puntos = [i for i in range(data.shape[1])]
        else:
            puntos = [min(range(len(wavelength)), key=lambda i: abs(wavelength[i] - num)) for num in selection]
        xlabel = 'Time (' + self._units['time_unit'] + ')'
        if plot_residues is False:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = ['_', ax]
        else:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 5]})
        fittes = self.results(fit_number=fit_number)
        if deconv:
            residues = data - fittes
        else:
            t0 = params['t0_1'].value
            index = np.argmin([abs(i - t0) for i in x])
            residues = data[index:, :] - fittes
        alpha, s = 0.80, 8
        for i in puntos:
            if plot_residues:
                ax[0].scatter(x, residues[:, i], marker='o', alpha=alpha, s=s)
            ax[1].scatter(x, data[:, i], marker='o', alpha=alpha, s=s)
            ax[1].plot(x, fittes[:, i], '-', color='r', alpha=0.5, lw=1.5)
            if len(puntos) <= 10:
                legenda = self._legend_plot_fit(data, wavelength, svd_fit, puntos)
                ax[1].legend(legenda, loc='best', ncol=1 if svd_fit else 2)
        if plot_residues:
            FiguresFormating.format_figure(ax[0], residues, x, size=size)
            ax[0].set_ylabel('Residues', size=size)
        FiguresFormating.format_figure(ax[1], data, x, size=size)
        FiguresFormating.axis_labels(ax[1], xlabel, r'$\Delta$A', size=size)
        plt.subplots_adjust(left=0.145, right=0.95)
        return fig, ax

    def DAS(self, number='all', fit_number=None):
        """
        returns an array of the Decay associated spectra. The number of rows is the number of species
        and the number of columns correspond to the wavelength vector.

        Parameters
        ----------
        number: list of inst or 'all'
            Defines the DAS spectra wanted, if there is tau_inf include -1 in the list
            e.g.: for a fit with three exponential, if the last two are wanted; number = [1, 2]
            e.g.2: the last two exponential plus tau_inf; number = [1, 2, -1]

        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered

        Returns
        ----------
        numpy 2d array
        """
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        if svd_fit and hasattr(self._fits[fit_number], 'params_svd'):
            params = self._fits[fit_number].params_svd
        values = [[params['pre_exp%i_' % (ii + 1) + str(i + 1)].value for i in range(data.shape[1])]
                  for ii in range(exp_no)]
        if deconv and tau_inf is not None:
            values.append([params['yinf_' + str(i + 1)].value for i in range(data.shape[1])])
        if number != 'all':
            assert type(number) == list, \
                'Number should be "all" or a list containing the desired species if tau inf include -1 in the list'
            das = self.DAS(fit_number=fit_number)
            wanted = self._wanted_DAS(exp_no, number, tau_inf)
            das = das[wanted, :]
        else:
            das = np.array(values)
        return das

    def plot_DAS(self, fit_number=None, number='all', precision=2, size=14, cover_range=None,
                 plot_integrated_DAS=False):
        """
        Function that generates a figure with the decay associated spectra (DAS) of the fit stored in
        the all_fit attributes
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered.
        
        number: list of inst or 'all'
            Defines the DAS spectra wanted, if there is tau_inf include -1 in the list
            e.g.: for a fit with three exponential, if the last two are wanted; number = [1, 2]
            e.g.2: the last two exponential plus tau_inf; number = [1, 2, -1]
        
        precision: int (default 2)
            Defines the number of decimal places of the legend legend
        
        size: int (default 14)
            size of the figure text labels including tick labels axis labels and legend
        
        cover_range: List of length 2 or None (default None)
            Defines a range in wavelength that will be cover in white. This can be use to cut the excitation
            wavelength range   
        
        plot_integrated_DAS: bool (default False)
            Defines in case if data has been derivate, to directly integrate the DAS
            
        Returns
        ----------
        Figure and axes matplotlib objects
        """
        # verify type of fit is: either fit to Singular vectors or global fit to traces
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        das = self.DAS(fit_number=fit_number)
        xlabel = self._get_wave_label_res(wavelength)
        legenda = self._legend_plot_DAS(params, exp_no, deconv, tau_inf, type_fit, precision)
        if number != 'all':
            wanted = self._wanted_DAS(exp_no, number, tau_inf)
            # constants=' '.join([str(i.split('=')[1]) for i in legenda[:number]])
            # print(f'WARNING! time constants of value {constants} has been used to fit')
            legenda = [legenda[i] for i in wanted]
        if type(derivative_space) == dict and plot_integrated_DAS:
            das = np.array([integral.cumtrapz(das[i, :], wavelength, initial=0) for i in range(len(das))])
        fig, ax = plt.subplots(1, figsize=(11, 6))
        for i in range(das.shape[0]):
            ax.plot(wavelength, das[i, :], label=legenda[i])
            plt.xlim(wavelength[0], wavelength[-1])
        leg = ax.legend(prop={'size': size})
        leg.set_zorder(np.inf)
        FiguresFormating.format_figure(ax, das, wavelength, x_tight=True, set_ylim=False)
        FiguresFormating.axis_labels(ax, xlabel, r'$\Delta$A', size=size)
        if cover_range is not None:
            FiguresFormating.cover_excitation(ax, cover_range, wavelength)
        return fig, ax

    def verify_fit(self, fit_number=None):
        """
        Function that generates a figure with a slider to evaluate every single trace fitted independently
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered.
            
        Returns
        ----------
        Figure and axes matplotlib objects
        """
        x, self._data_fit, self._wavelength_fit, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        xlabel = 'Time (' + self._units['time_unit'] + ')'
        self._fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 5]})
        self._fittes = self.results(fit_number=None)
        if deconv:
            self._residues = self._data_fit - self._fittes
            self._x_verivefit = x * 1.0
        else:
            t0 = params['t0_1'].value
            index = np.argmin([abs(i - t0) for i in x])
            self._x_verivefit = x[index:]
            self._residues = self._data_fit[index:, :] - self._fittes
        initial_i = self._data_fit.shape[1] // 5
        self._l = ax[1].plot(self._x_verivefit, self._data_fit[:, initial_i], marker='o', ms=3, linestyle=None,
                             label='raw data')[0]
        self._lll = ax[0].plot(self._x_verivefit, self._residues[:, initial_i], marker='o', ms=3, linestyle=None,
                               label='residues')[0]
        self._ll = ax[1].plot(self._x_verivefit, self._fittes[:, initial_i], alpha=0.5, lw=1.5, color='r', label='fit')[0]
        delta_f = 1.0
        _, maxi = self._data_fit.shape
        axcolor = 'orange'
        axspec = self._fig.add_axes([0.20, .05, 0.60, 0.02], facecolor=axcolor)
        self._sspec = Slider(axspec, 'curve number', 1, maxi, valstep=delta_f, valinit=maxi//5)
        self._sspec.on_changed(self._update_verified_Fit)
        FiguresFormating.format_figure(ax[0], self._residues, self._x_verivefit, size=14)
        ax[0].set_ylabel('Residues', size=14)
        FiguresFormating.format_figure(ax[1], self._data_fit, self._x_verivefit, size=14)
        ax[1].legend(loc='upper right')
        ax[0].legend(loc='upper right')
        title = round(self._wavelength_fit[initial_i])
        self._title = ax[0].set_title(f'{title} nm')
        plt.title(f'{title} nm')
        FiguresFormating.axis_labels(ax[1], xlabel, r'$\Delta$A', size=14)
        # self._fig.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        return self._fig, ax

    def _update_verified_Fit(self, val):
        """
        updates the verified_fit function
        """
        # amp is the current value of the slider
        value = self._sspec.val
        value = int(round(value))
        # update curve
        title = round(self._wavelength_fit[value])
        self._title.set_text(f'{title} nm')
        plt.title(f'{title} nm')
        self._l.set_ydata(self._data_fit[:, value])
        self._ll.set_ydata(self._fittes[:, value])
        self._lll.set_ydata(self._residues[:, value])
        # redraw canvas while idle
        self._fig.canvas.draw_idle()

    def plot_concentrations(self, fit_number=None, names=None, plot_total_c=True, legend=True,  size=14,):  # tmp function.
        """
        Function that generates a figure with the concentration evolution of a Target fit
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered.
        
        names: list (default None)
            List that allows to redefine the names of the components. 
            Is none the default names "specie" is use.
            The names are display in the legend if legend is True
        
        plot_total_c: bool (default True)
            Defines if the Total concentration is 
        
        legend: bool (default True)
            Defines if the legend is display
        
        size: int (default 14)
            size of the figure text labels including tick labels axis labels and legend
            
        Returns
        ----------
        Figure and axes matplotlib objects
        """
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)

        if type_fit == 'Exponential':
            msg = 'This function is only available for target fit'
            raise ExperimentException(msg)
        xlabel = 'Time (' + self._units['time_unit'] + ')'
        maxi_tau = -1 / params['k_%i%i' % (exp_no - 1, exp_no - 1)].value
        if maxi_tau > x[-1]:
            maxi_tau = x[-1]
        # size of the matrix = no of exponenses = no of species
        coeffs, eigs, eigenmatrix = solve_kmatrix(exp_no, params)
        t0 = params['t0_1'].value
        fwhm = params['fwhm_1'].value/2.35482
        expvects = [coeffs[i] * ModelCreator.expGauss(x - t0, -1/eigs[i], fwhm)
                    for i in range(len(eigs))]
        concentrations = [sum([eigenmatrix[i, j] * expvects[j]
                               for j in range(len(eigs))])
                          for i in range(len(eigs))]
        if names is None or len(names) != exp_no:
            names = [f'Specie {i}' for i in range(exp_no)]
        fig, ax = plt.subplots(1, figsize=(8, 6))
        for i in range(len(eigs)):
            ax.plot(x, concentrations[i], label=names[i])
        if plot_total_c:
            allc = sum(concentrations)
            ax.plot(x, allc, label='Total concentration')  # sum of all for checking => should be unity
        if legend:
            plt.legend(loc='best')
        FiguresFormating.format_figure(ax, concentrations, x, x_tight=True, set_ylim=False)
        FiguresFormating.axis_labels(ax, xlabel, 'Concentration (A.U.)', size=size)
        plt.xlim(-3, round(maxi_tau * 7))
        return fig, ax
    
    def print_results(self, fit_number=None):
        """
        Print out a summarize result of the fit.
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None the last fit in  will
            be considered.
        """
        if fit_number is None:
            fit_number = max(self._fits.keys())
        _, data, _, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        if deconv:
            names=['t0_1', 'fwhm_1']+['tau%i_1'%(i+1) for i in range(exp_no)]
            print_names = ['time 0', 'fwhm']
        else:
            names=['t0_1']+['tau%i_1'%(i+1) for i in range(exp_no)]
            print_names = ['time 0']
        print_names = print_names + ['tau %i' %i for i in range(1,exp_no + 1)] 
        # print_resultados='\t'+',\n\t'.join([f'{name.split("_")[0]}:\t{round(params[name].value,4)}\t{params[name].vary}' for name in names])
        print(f'Fit number {fit_number}: \tGlobal {type_fit} fit')
        print('-------------------------------------------------')
        print('Results:\tParameter\t\tInitial value\tFinal value\t\tVary')
        for i in range(len(names)):
            line = [f'\t{print_names[i]}:', '{:.4f}'.format(params[names[i]].init_value),
                      '{:.4f}'.format(params[names[i]].value), f'{params[names[i]].vary}']
            print('\t\t'+'   \t\t'.join(line))
        print('Details:')
        if svd_fit:
            trace, avg = 'Nº of singular vectors', '0'
        else:
            trace = 'Nº of traces'
            avg = self._fits[fit_number].details["avg_traces"]
        print(f'\t\t{trace}: {data.shape[1]}; average: {avg}')
        if type_fit == 'Exponential':
            print(f'\t\tFit with {exp_no} exponential; Deconvolution {deconv}')
            print(f'\t\tTau inf: {tau_inf}')
        print(f'\t\tNumber of iterations: {self._fits[fit_number].nfev}')
        print(f'\t\tNumber of parameters optimized: {len(params)}')
        print(f'\t\tWeights: {self._fits[fit_number].weights}')
        
    def _get_wave_label_res(self, wavelength):
        """
        Returns a formatted string from the units attribute
        """
        if wavelength is None:
            xlabel = 'pixel'
        elif self._units['wavelength_unit'] == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self._units["wavelength_unit"]})'
        return xlabel

    def _legend_plot_DAS(self, params, exp_no, deconv, tau_inf, type_fit, precision):
        """
        returns legend for plot_DAS function
        """
        legenda = [self._unit_formater.value_formated(params['tau%i_1' % (i + 1)].value, precision)
                   for i in range(exp_no)]
        if deconv and type_fit == 'Exponential':
            if tau_inf is None:
                pass
            elif tau_inf != 1E+12:
                legenda.append(self._unit_formater.value_formated(tau_inf, precision))
            else:
                legenda.append(r'$\tau$ = inf')
        return legenda

    def _legend_plot_fit(self, data, wavelength, svd_fit, puntos):
        """
        returns legend for plot_fit function in case the number of fits are less or equal to 10
        """
        if wavelength is None:
            wavelength = np.array([i for i in range(len(data[1]))])
        if svd_fit:
            legend = ['_' for i in range(data.shape[1])] + ['left SV %i' % i for i in
                                                                 range(1, data.shape[1] + 1)]
        elif wavelength is not None:
            val = 'cm$^{-1}$' if self._units['wavelength_unit'] == 'cm-1' else self._units['wavelength_unit']
            legend = ['_' for i in range(len(puntos))] + [f'{round(wavelength[i])} {val}' for i in puntos]
        else:
            legend = ['_' for i in range(len(puntos))] + [f'curve {i}' for i in range(data.shape[1])]
        return legend

    def _wanted_DAS(self, exp_no, number, tau_inf):
        """
        return a sub-array of DAS
        """
        posible = [i + 1 for i in range(exp_no)]
        if tau_inf is not None:
            posible.append(-1)
        wanted = [ii for ii, i in enumerate(posible) if i in number]
        return wanted

    def _get_values(self, fit_number=None, verify_svd_fit=False):
        """
        return values from the results object
        """
        if fit_number is None:
            fit_number = max(self._fits.keys())
        params = self._fits[fit_number].params
        data = self._fits[fit_number].data
        x = self._fits[fit_number].x
        svd_fit = self._fits[fit_number].details['svd_fit']
        wavelength = self._fits[fit_number].wavelength
        deconv = self._fits[fit_number].details['deconv']
        tau_inf = self._fits[fit_number].details['tau_inf']
        exp_no = self._fits[fit_number].details['exp_no']
        derivative_space = self._fits[fit_number].details['derivate']
        # check for type of fit done target or exponential
        type_fit = self._fits[fit_number].details['type']
        return x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space