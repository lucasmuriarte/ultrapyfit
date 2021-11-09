# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:35:41 2020

@author: lucas
"""
from ultrafast.utils.divers import FiguresFormating, solve_kmatrix, TimeUnitFormater
from ultrafast.utils.Preprocessing import ExperimentException
from ultrafast.graphics.styles.set_styles import use_style
import matplotlib.pyplot as plt
import numpy as np
from ultrafast.fit.ModelCreator import ModelCreator
import scipy.integrate as integral
from matplotlib.widgets import Slider


class ExploreResults:
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

    def get_gloabl_fit_curve_results(self, fit_number=None, verify_svd_fit=False):
        """
        Returns a data set created from the best parameters values.

        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None
            the last fit in  will be considered

        verify_svd_fit: bool (default  False)
            If True it will return the single fit perform to every trace of the
            spectra after an svd fit. If false and the fit is an SVD, the
            values return are the fit to the svd left vectors.
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

        elif type_fit == 'Exponential convolved':
            curve_resultados = data * 0.0
            # t0 = result_params['t0_1'].value
            # index = np.argmin([abs(i - t0) for i in x])
            # curve_resultados = 0.0 * data[index:, :]
            for i in range(nx):
                curve_resultados[:, i] = model.expNDatasetIRF(result_params,
                                                              i, deconv)

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

    @use_style
    def plot_global_fit(self, fit_number=None, selection=None,
                        plot_residues=True, style='lmu_res', ):
        """
        Function that generates a figure with the results of the fit stored in
        the all_fit attributes.  If less than 10 traces are fitted or selected
        a legend will be displayed
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None
            the last fit in  will be considered
        
        selection: list or None (default None)    
            If None all the traces fitted will be plotted, if not only those
            selected in the lis
        
        plot_residues: Bool (default True)
            If True the Figure returned will contain two axes a top one with
            the residues, and the bottom one with the fit and data
            
        style: style valid name (default 'lmu_res')
            defines the style to format the output figure, it can be any defined
            matplotlib style, any ultrafast style (utf) or any user defined
            style that follows matplotlib or utf styles and is saved in the
            correct folder. Check styles for more information.
        
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
        xlabel = f'Time ({self.time_unit})'
        if plot_residues is False:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = ['_', ax]
        else:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                   gridspec_kw={'height_ratios': [1, 5]})
        fittes = self.get_gloabl_fit_curve_results(fit_number=fit_number)
        alpha, s = 0.80, 8
        if type(deconv) == bool:
            if deconv:
                residues = data - fittes
                x_residues = x*1.0
            else:
                t0 = params['t0_1'].value
                index = np.argmin([abs(i - t0) for i in x])
                residues = data[index:, :] - fittes
                x_residues = x[index:]
        else:
            residues = data - fittes
            x_residues = x * 1.0
            # t0 = params['t0_1'].value
            # index = np.argmin([abs(i - t0) for i in x])
            # residues = data[index:, :] - fittes
            # x_residues = x[index:]
            ax[1].scatter(x, deconv, marker='o', alpha=alpha, s=s, c='k')
        for i in puntos:
            if plot_residues:
                ax[0].scatter(x_residues, residues[:, i], marker='o', alpha=alpha, s=s)
            ax[1].scatter(x, data[:, i], marker='o', alpha=alpha, s=s)
            ax[1].plot(x_residues, fittes[:, i], '-', color='r', alpha=0.5, lw=1.5)
            if len(puntos) <= 10:
                legenda = self._legend_plot_fit(data, wavelength, svd_fit, puntos)
                ax[1].legend(legenda, loc='best', ncol=1 if svd_fit else 2)
        if plot_residues:
            # FiguresFormating.format_figure(ax[0], residues, x, size=size)
            ax[0].set_ylabel('Residues')
        # iguresFormating.format_figure(ax[1], data, x)
        FiguresFormating.axis_labels(ax[1], xlabel, r'$\Delta$A')
        plt.subplots_adjust(left=0.145, right=0.95)
        return fig, ax

    def get_DAS(self, number='all', fit_number=None, convert_to_EAS=False):
        """
        returns an array of the Decay associated spectra. The number of rows is
        the number of species and the number of columns correspond to the
        wavelength vector.

        Parameters
        ----------
        number: list of inst or 'all'
            Defines the DAS spectra wanted, if there is tau_inf include -1 in
            the list:
            e.g.: for a fit with three exponential, if the last two are wanted;
                   number = [1, 2]
            e.g.2: the last two exponential plus tau_inf; number = [1, 2, -1]

        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary.
            If None the last fit in  will be considered
        
        convert_to_EAS:
            return the especies associted spectra, obtained as a linear 
            combination of the DAS and considering a sequential model.
            
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
        if type(deconv) == bool:
            if deconv and tau_inf is not None:
                values.append([params['yinf_' + str(i + 1)].value for i in range(data.shape[1])])
            elif not deconv:
                values.append([params['y0_' + str(i + 1)].value for i in
                               range(data.shape[1])])
        das = np.array(values)
        if convert_to_EAS:
            das = self.das_to_eas(das, params, wavelength,
                                  exp_no, tau_inf, deconv)
        if number != 'all':
            msg = 'Number should be "all" or a list containing the desired ' \
                  'species if tau inf include -1 in the list'
            assert type(number) == list, msg
            wanted = self._wanted_DAS(exp_no, number, tau_inf)
            das = das[wanted, :]

        return das

    @use_style
    def plot_DAS(self, fit_number=None, number='all', plot_offset=True,
                 precision=2, style='lmu_spec',
                 cover_range=None,
                 plot_integrated_DAS=False,
                 convert_to_EAS=False):
        """
        Function that generates a figure with the decay associated spectra (DAS)
         of the fit stored in the all_fit attributes
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None
            the last fit in  will be considered.
        
        number: list of inst or 'all' (default 'all')
            Defines the DAS spectra wanted, if there is tau_inf include -1 in
            the list (Note we start counting by '0', thus for tau1 '0' should be
            pass):
            e.g.: for a fit with three exponential, if the last two are wanted;
                  number = [1, 2]
            e.g.2: the last two exponential plus tau_inf; number = [1, 2, -1]

        plot_offset: bool (default True)
            If True the offset of a global fit without deconvolution will be
            plot. (Only applicable for exponential or target fit without
            deconvolution)

        precision: int (default 2)
            Defines the number of decimal places of the legend legend
        
        style: style valid name (default 'lmu_spec')
            defines the style to format the output figure, it can be any defined
            matplotlib style, any ultrafast style (utf) or any user defined
            style that follows matplotlib or utf styles and is saved in the
            correct folder. Check styles for more information.
        
        cover_range: List of length 2 or None (default None)
            Defines a range in wavelength that will be cover in white. This can
            be use to cut the excitation wavelength range
        
        plot_integrated_DAS: bool (default False)
            Defines in case if data has been derivative, to directly integrate
            the DAS.
        
        convert_to_EAS:
            return the especies associted spectra, obtained as a linear 
            combination of the DAS and considering a sequential model.    
        
        Returns
        ----------
        Figure and axes matplotlib objects
        """
        # verify type of fit is: either fit to Singular vectors or global fit to traces
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        das = self.get_DAS(number=number, fit_number=fit_number,
                           convert_to_EAS=convert_to_EAS)

        xlabel = self._get_wave_label_res(wavelength)
        legenda = self._legend_plot_DAS(params, exp_no, deconv, tau_inf, type_fit, precision)

        # check DAS that are selected and adapt the legend
        if number != 'all':
            wanted = self._wanted_DAS(exp_no, number, tau_inf)
            # print a warning with elements not being plotted
            select = [legenda[i] for i in wanted]
            constants = ', '.join([i for i in legenda if i not in select])
            msg = f'WARNING! the components: {constants}; ' \
                  f'have been used to fit the data and are not been displayed'
            print(msg)
            legenda = [legenda[i] for i in wanted]

        # integrate the DAS in case fit is done in derivate data
        if derivative_space and plot_integrated_DAS:
            das = np.array([integral.cumtrapz(das[i, :], wavelength, initial=0)
                            for i in range(len(das))])

        fig, ax = plt.subplots(1, figsize=(11, 6))
        n_das = das.shape[0]

        for i in range(n_das):
            # if to decide if to plot the offset or not
            if i == n_das-1 and not deconv and not plot_offset:
                pass
            else:
                ax.plot(wavelength, das[i, :], label=legenda[i])
        plt.xlim(wavelength[0], wavelength[-1])
        leg = ax.legend()
        leg.set_zorder(np.inf)
        # FiguresFormating.format_figure(ax, das, wavelength, x_tight=True, set_ylim=False)
        FiguresFormating.axis_labels(ax, xlabel, r'$\Delta$A')
        if cover_range is not None:
            FiguresFormating.cover_excitation(ax, cover_range, wavelength)
        return fig, ax

    def das_to_eas(self, das, params, wavelength, n_exp, tau_inf, deconv):
        """
        Recalculates DAS spectra into EAS, treating y_inf as final product.
        Replaces DAS with EAS, so one should treat this data as EAS after
        calling this method.
        It works with matrices and should be correct for any number of exp's.
        """

        no_of_wavelengths = wavelength.shape[0]
        tau_inf_enabled = tau_inf  # set here if it is enabled
        no_of_exps = int(n_exp)  # load here number of exponentials
        # params = self.params.copy()
        if not deconv:
            offset = das[-1, :]
            das = das[:-1, :]
        eas = np.zeros(das.shape)
        ##########
        if tau_inf_enabled:
            size_of_kmatrix = no_of_exps + 1
        else:
            size_of_kmatrix = no_of_exps


        # build kmatrix of the sequential model
        kmatrix = np.zeros((size_of_kmatrix, size_of_kmatrix))
        k_values = []
        for i in range(no_of_exps):
            k_value = 1 / params['tau%i_' % (i + 1) + str(1)].value
            kmatrix[i, i] = -k_value
            if (i + 1 < size_of_kmatrix):
                kmatrix[i + 1, i] = k_value
            k_values.append(k_value)

        # initialize initial concentration values,
        c_initials = np.zeros(size_of_kmatrix)
        # since it's sequential only the first is 1
        c_initials[0] = 1.0

        # now extract eigenvalues and eigenvectors, to be able to reduce kmatrix
        # to the only-diagonal form. then we can easily differentiate and get solution
        eigs_out, vects_out = np.linalg.eig(kmatrix)

        # note that eigenvalues can come out not ordered. we want them in the
        # same order and values as k1,k2,k3... note that the same values are
        # obtained only if kmatrix describes sequential model. more complicated
        # models can give different k values than these used to build kmatrix
        if tau_inf_enabled and deconv:
            oryginal_k_order = -np.array(k_values + [0.0])
        else:
            oryginal_k_order = -np.array(k_values)

        ks_ordering = np.argsort(oryginal_k_order)
        ks_reverse_ordering = np.argsort(ks_ordering)

        # sort eigenthings
        sort_ordering = np.argsort(eigs_out)
        eigs_sorted = eigs_out[sort_ordering]
        vects_sorted = vects_out[:, sort_ordering]

        # order eigenthings like k1,k2,k3....
        eigs = eigs_sorted[ks_reverse_ordering]
        vects = vects_sorted[:, ks_reverse_ordering]

        # then solve linear equation, where t=0 so you have
        # eigvects_matrix*vect_of_concentrations = vect_of_initial_values
        # by this you get coeffs which are before diagonalized exp functions
        # print(vects.shape)
        # print(c_initials.shape)
        coeffs = np.linalg.solve(vects, c_initials)

        # ok, now you make diagonal array with these coeffs
        em_matrix = np.identity(coeffs.shape[0]) * np.transpose(
            coeffs[np.newaxis])
        # and you multiply eigenvector matrix by this. so you have:
        # fit_of_data = eas_array * d_matrix * exp_matrix
        # fit_of_data = das_array * exp_matrix
        d_matrix = np.dot(vects, em_matrix)

        # now you can just transpose that, and by comparison of das and sas,
        # you can get eas values from linear equation:
        d_t = np.transpose(d_matrix)
        # like below, but need to do this in loop for all kinetics:
        # EASv = np.linalg.solve(d_t, DASv)

        # idea is to iterate slowly over every kinetic, and change exp-associated
        # preexp factors into species associated preexp factors
        for wavelength_num in range(no_of_wavelengths):
            DASv = das[:, wavelength_num]
            # DASv = [params[
            #        'pre_exp%i_%i' % (i + 1, wavelength_num + 1)].value
            #        for i in range(no_of_exps)]
            # if tau_inf_enabled:
            #     DASv.append(
            #         self.params['yinf_%i' % (wavelength_num + 1)].value)


            EASv = np.linalg.solve(d_t, np.array(DASv))
            eas[:, wavelength_num] = EASv
            # lets replace das with eas:
            # for i in range(no_of_exps):
            #    params['pre_exp%i_' % (i + 1) + str(wavelength_num + 1)].set(
            #        value=EASv[i])
            # if (tau_inf_enabled):
            #    params['yinf_' + str(wavelength_num + 1)].set(
            #        value=EASv[no_of_exps])
        if not deconv:
            # add offset (vertical stacking)
            eas = np.r_[eas, [offset]]
        return eas
        # lets put back the params, but now pre_exps are EAS, not DAS!
        # self.params = params

    def plot_verify_fit(self, fit_number=None):
        """
        Function that generates a figure with a slider to evaluate every single
        trace fitted independently.
        
        Parameters
        ----------
        fit_number: int or None (default None)
            defines the fit number of the results all_fit dictionary. If None
            the last fit in  will be considered.
            
        Returns
        ----------
        Figure and axes matplotlib objects
        """
        x, self._data_fit, self._wavelength_fit, params, exp_no, deconv, tau_inf, svd_fit, type_fit, derivative_space = \
            self._get_values(fit_number=fit_number)
        xlabel = f'Time ({self.time_unit})'
        self._fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 5]})
        self._fittes = self.get_gloabl_fit_curve_results(fit_number=None)
        self._x_verivefit = x * 1.0
        if type(deconv) == bool:
            if not deconv:
                t0 = params['t0_1'].value
                index = np.argmin([abs(i - t0) for i in x])
                self._x_verivefit_residues = x[index:]
                self._residues = self._data_fit[index:, :] - self._fittes
        else:
            self._residues = self._data_fit - self._fittes
            self._x_verivefit_residues = x * 1.0
        initial_i = self._data_fit.shape[1] // 5
        self._l = ax[1].plot(self._x_verivefit, self._data_fit[:, initial_i], marker='o', ms=3, linestyle=None,
                             label='raw data')[0]
        self._lll = ax[0].plot(self._x_verivefit_residues, self._residues[:, initial_i], marker='o', ms=3, linestyle=None,
                               label='residues')[0]
        self._ll = ax[1].plot(self._x_verivefit_residues, self._fittes[:, initial_i], alpha=0.5, lw=1.5, color='r', label='fit')[0]
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
        xlabel = f'Time ({self.time_unit})'
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
    
    def print_fit_results(self, fit_number=None):
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
        fit = self._fits[fit_number]
        print(f"Fit number {fit_number}: \t" + fit.__str__())
        
    def _get_wave_label_res(self, wavelength):
        """
        Returns a formatted string from the units attribute
        """
        if wavelength is None:
            xlabel = 'pixel'
        elif self.wavelength_unit == 'cm-1':
            xlabel = 'Wavenumber (cm$^{-1}$)'
        else:
            xlabel = f'Wavelength ({self.wavelength_unit})'
        return xlabel

    def _legend_plot_DAS(self, params, exp_no, deconv, tau_inf, type_fit, precision):
        """
        returns legend for plot_DAS function
        """
        legenda = [self._unit_formater.value_formated(
            abs(params['tau%i_1' % (i + 1)].value), precision)
            for i in range(exp_no)]
        if deconv and type_fit == 'Exponential':
            if tau_inf is None:
                pass
            elif tau_inf != 1E+12:
                legenda.append(self._unit_formater.value_formated(tau_inf,
                                                                  precision))
            else:
                legenda.append(r'$\tau$ = inf')
        elif not deconv:
            legenda.append(r'Offset')
        return legenda

    def _legend_plot_fit(self, data, wavelength, svd_fit, puntos):
        """
        returns legend for plot_fit function in case the number of fits are
        less or equal to 10
        """
        if wavelength is None:
            wavelength = np.array([i for i in range(len(data[1]))])
        if svd_fit:
            legend = ['_' for i in range(data.shape[1])] + \
                     ['left SV %i' % i for i in range(1, data.shape[1] + 1)]
        elif wavelength is not None:
            if self.wavelength_unit == 'cm-1':
                val = 'cm$^{-1}$'
            else:
                val =self.wavelength_unit
            legend = ['_' for i in range(len(puntos))] + \
                     [f'{round(wavelength[i])} {val}' for i in puntos]
        else:
            legend = ['_' for i in range(len(puntos))] + \
                     [f'curve {i}' for i in range(data.shape[1])]
        return legend

    @staticmethod
    def _wanted_DAS(exp_no, number, tau_inf):
        """
        return a list of numbers equivalent to the sub-array of DAS wanted.
        Note counting starts at 0.
        """
        posible = [i for i in range(exp_no)]
        posible.append(-1)
        wanted = [posible[ii] for ii, i in enumerate(posible) if i in number]
        return wanted

    def _get_values(self, fit_number=None, verify_svd_fit=False):
        """
        return values from the results object
        """
        if fit_number is None:
            fit_number = max(self._fits.keys())
        fit = self._fits[fit_number]
        return fit.get_values()