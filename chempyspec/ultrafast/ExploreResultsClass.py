# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:35:41 2020

@author: lucas
"""
from chempyspec.ultrafast.outils import FiguresFormating, solve_kmatrix
import matplotlib.pyplot as plt
import numpy as np
from ModelCreatorClass import ModelCreator
import scipy.integrate as integral
from matplotlib.widgets import Slider


class ExploreResults(FiguresFormating):
    def __init__(self, fits, **kwargs):
        units = dict({'time_unit': 'ps', 'time_unit_high': 'ns', 'time_unit_low': 'fs', 'wavelength_unit': 'nm',
                      'factor_high': 1000, 'factor_low': 1000}, **kwargs)
        if type(fits) == dict:
            self.all_fit = fits
        else:
            self.all_fit = {1: fits}
        self.units = units
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

    def results(self, fit_number=None, verify_svd_fit=False):
        """
        Returns a data set created from the best parameters values.

         Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results object. If None the last fit in the result object will
            be considered

        verify_svd_fit: bool (default  False)
            If True it will return the single fit perform to every trace of the spectra after an svd fit
            If false and the fit is an SVD, the values retunr are the fit to the svd
            If is not an SVD fit this parameter is not applicable
        """
        x, data, wavelength, result_params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space,  = \
            self._get_values(fit_number=fit_number, verify_svd_fit=verify_svd_fit)
        model = ModelCreator(x, exp_no, tau_inf)
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
                curve_resultados[:, i] = model.expNGaussDatasetTM(result_params, i, [coeffs, eigs, eigenmatrix])
        return curve_resultados

    def plot_fit(self, size=14, fit_number=None, selection=None, plot_residues=True):
        """

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
        xlabel = 'Time (' + self.units['time_unit'] + ')'
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
        #        plt.axhline(linewidth=1,linestyle='--', color='k')
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
            e.g.2: the last two exponentials plus tau_inf; number = [1, 2, -1]

        fit_number: int or None (default None)
            defines the fit number of the results object. If None the last fit in the result object will
            be considered

        Returns
        ----------
        numpy 2d array
        """
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        values = [[params['pre_exp%i_' % (ii + 1) + str(i + 1)].value for i in range(data.shape[1])]
                  for ii in range(exp_no)]
        if deconv and tau_inf is not None:
            values.append([params['yinf_' + str(i + 1)].value for i in range(data.shape[1])])
        if number is not 'all':
            assert type(number) == list, \
                'Number should be "all" or a list containing the desired species if tau inf include -1 in the list'
            das = self.DAS(fit_number=fit_number)
            wanted = self._wanted_DAS(exp_no, number, tau_inf)
            das = das[wanted, :]
        else:
            das = np.array(values)
        return das

    def plotDAS(self, number='all', fit_number=None, size=14, precision=2, plot_integrated_DAS=False, cover_range=None):
        """
        REtunrs the Decay Asociated Spectra of the FIt
        Parameters: 
            number --> 'all' or a list conataining the decay of the species you want, plus -1 if you want the tau inf if existing
                        eg1.: if you want the decay of the second and third, [2,3] // eg2.:  fist third and inf, [2,3,-1]
                            
        """
        # verify type of fit is: either fit to Singular vectors or global fit to traces
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        das = self.DAS(fit_number=fit_number)
        xlabel = self._get_wave_label(wavelength)
        legenda = self._legend_plot_DAS(params, exp_no, deconv, tau_inf, type_fit, precision)
        if number is not 'all':
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

    def verifiedFit(self, fit_number=None):
        x, self._data_fit, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        xlabel = 'Time (' + self.units['time_unit'] + ')'
        self._fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 5]})
        self._fittes = self.results(params)
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
                             label='raw data')
        self._lll = ax[0].plot(self._x_verivefit, self._residues[:, initial_i], marker='o', ms=3, linestyle=None,
                               label='residues')
        self._ll = ax[1].plot(self._x_verivefit, self._fittes[:, initial_i], alpha=0.5, lw=1.5, color='r', label='fit')
        delta_f = 1.0
        _, maxi = self._data_fit.shape
        axcolor = 'orange'
        axspec = self._fig.add_axes([0.20, .02, 0.60, 0.01], facecolor=axcolor)
        self._sspec = Slider(axspec, 'curve number', 0, maxi - 1, valstep=delta_f, valinit=0)
        self._sspec.on_changed(self._update_verified_Fit)
        FiguresFormating.format_figure(ax[0], self._residues, self._x_verivefit, size=14)
        ax[0].set_ylabel('Residues', size=14)
        FiguresFormating.format_figure(ax[1], self._data_fit, self._x_verivefit, size=14)
        ax[1].legend(loc='upper right')
        ax[0].legend(loc='upper right')
        title = round(self._wavelength_fit[initial_i])
        plt.title(f'{title} nm')
        FiguresFormating.axis_labels(ax, xlabel, r'$\Delta$A', size=14)
        self._fig.tight_layout()
        plt.subplots_adjust(bottom=0.25)
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
        plt.title(f'{title} nm')
        self._l.set_ydata(self._data_fit[:, value])
        self._ll.set_ydata(self._fittes[:, value])
        self._lll.set_ydata(self._residues[:, value])
        # redraw canvas while idle
        self._fig.canvas.draw_idle()

    def plotConcentrations(self, fit_number=None, size=14, names=None, plot_total_c=True, legend=True):  # tmp function.
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)

        if type_fit == 'Expeonential':
            return 'This fucntion is only available for target fit'
        else:
            xlabel = 'Time (' + self.units['time_unit'] + ')'
            maxi_tau = -1 / params['k_%i%i' % (exp_no - 1, exp_no - 1)].value
            if maxi_tau > x[-1]:
                maxi_tau = x[-1]
            # size of the matrix = no of exponenses = no of species
            coeffs, eigs, eigenmatrix = solve_kmatrix(exp_no, params)
            t0 = params['t0_1'].value
            fwhm = params['fwhm_1'].value
            expvects = [coeffs[i] * ModelCreator.expGauss(x - t0, -eigs[i], fwhm / 2.35482) for i in range(len(eigs))]
            concentrations = [sum([eigenmatrix[i, j] * expvects[j] for j in range(len(eigs))]) for i in
                              range(len(eigs))]
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

    def _get_wave_label(self, wavelength):
        """
        Returns a formatted string from the units attribute
        """
        if wavelength is None:
            xlabel = 'pixel'
        elif self.units['wavelength_unit'] == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self.units["wavelength_unit"]})'
        return xlabel

    def _legend_plot_DAS(self, params, exp_no, deconv, tau_inf, type_fit, precision):
        """
        returns legend for plot_DAS function
        """
        legenda = []
        for i in range(exp_no):
            tau = params['tau%i_1' % (i + 1)].value
            if tau < 0.09:
                tau *= self.units['factor_low']
                legenda.append(
                    rf'$\tau {i + 1}$ = ' + '{:.2f}'.format(round(tau, precision)) + ' ' + self.units['time_unit_low'])
            elif tau > 999:
                if tau > 1E12:
                    legenda.append(r'$\tau$ = inf')
                else:
                    tau /= self.units['factor_high']
                    legenda.append(
                        rf'$\tau {i + 1}$ = ' + '{:.2f}'.format(round(tau, precision)) + ' '
                        + self.units['time_unit_high'])
            else:
                legenda.append(rf'$\tau {i + 1}$ = ' + '{:.2f}'.format(round(tau, precision)) + ' '
                               + self.units['time_unit'])
        if deconv and type_fit == 'Exponential':
            if tau_inf is None:
                pass
            elif tau_inf != 1E+12:
                legenda.append(r'$\tau$ = {:.2f}'.format(round(tau_inf / self.units['factor_high'], precision))
                               + ' ' + self.units['time_unit_high)'])
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
            legend = ['_' for i in range(data.shape[1] + 1)] + ['left SV %i' % i for i in
                                                                 range(1, data.shape[1] + 1)]
        elif wavelength is not None:
            legend = ['_' for i in range(len(puntos) + 1)] + [f'{round(wavelength[i])} nm' for i in puntos]
        else:
            legend = ['_' for i in range(len(puntos) + 1)] + [f'curve {i}' for i in range(data.shape[1])]
        return legend

    def _wanted_DAS(self, exp_no, number, tau_inf):
        posible = [i + 1 for i in range(exp_no)]
        if tau_inf is not None:
            posible.append(-1)
        wanted = [ii for ii, i in enumerate(posible) if i in number]
        return wanted

    def _get_values(self, fit_number=None, verify_svd_fit=False):
        """
        return values from the results object
        """
        if fit_number is None and verify_svd_fit:
            fit_number = max(self.all_fit.keys())
        if self.all_fit[fit_number].details['svd_fit']:
            params = self.all_fit[fit_number][11]
            data = self.all_fit[fit_number][10]
        else:
            params = self.all_fit[fit_number].params
            data = self.all_fit[fit_number].data
        x = self.all_fit[fit_number].x
        svd_fit = self.all_fit[fit_number].details['svd_fit']
        wavelength = self.all_fit[fit_number].wavelength
        deconv = self.all_fit[fit_number].details['deconv']
        tau_inf = self.all_fit[fit_number].details['tau_inf']
        exp_no = self.all_fit[fit_number].details['exp_no']
        derivative_space = self.all_fit[fit_number].details['derivate']
        # check for type of fit done target or exponential
        type_fit = self.all_fit[fit_number].details['type']
        return x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space