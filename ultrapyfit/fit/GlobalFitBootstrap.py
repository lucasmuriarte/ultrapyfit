import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from ultrapyfit.utils.Preprocessing import ExperimentException
from ultrapyfit.graphics.ExploreResults import ExploreResults
from ultrapyfit.fit.GlobalFit import GlobalFitExponential
from ultrapyfit.fit.GlobalFit import GlobalFitTarget
from seaborn import histplot, kdeplot
from matplotlib.offsetbox import AnchoredText
import concurrent


class BootStrap:
    """
    Class used to calculate confidence intervals (CI) from a bootstrap of
    either the residues or the original data to obtain a more robust CI
    value than those obtain by inverting the  the second derivative error
    matrix. The bootstrap calculations takes long, since we recommend a minimum
    of 200 bootstrap fit samples to have an correct estimation of the parameters
    CI and correlations between the parameters.
    Attributes
    ----------
    fit_results: lmfit results Object
        Should be an lmfit result object obtained either from
        GlobalFitTarget, GlobalFitExponential or Experiment classes

    bootstrap_result: pandas dataFrame (default None)
        Pandas data frame containing where the bootstrap results are appended
        initially can be None, which will imply creating a new one from zero.
        Alternatively the results of a previous bootstrap may be passed to
        increase the number of analysis.

    data_simulated: string
        contain the type of data sets simulated: "data" if they are simulated by
        random shuffling the data, or residues if they are obtained by random
        shuffling a part of the residuals.

    confidence_interval: lmfit parameters object
        contains the confidence interval for the decay times calculated from the
        bootstrap_result dataFrame

    datasets: numpy array
        contain the simulated data sets for fitting produced either directly
        from the sample or from the residues.

    fitter: GlobalFitTarget / GlobalFitExponential
        Contains the fitter used to obtained the fit_results passed

    params: lmfit Parameter object
        Contains the parameters used to obtained the fit_results passed
    """

    def __init__(self, fit_results, bootstrap_result=None,
                 workers=2, time_unit='ps'):
        """
        constructor function:
        Parameters
        ----------
        fit_results: lmfit results Object
            Should be an lmfit result object obtained either from
            GlobalFitTarget, GlobalFitExponential or Experiment classes

        bootstrap_result: pandas dataFrame (default None)
            Pandas data frame containing where the bootstrap results are
            appended initially can be None, which will imply creating a new one
            from zero. Alternatively the results of a previous bootstrap may be
            passed to increase the number of analysis.

        workers: int (default 2)
            number of workers (CPU cores) that will be used if the fit fit is
            run in parallel. Recommend to used as maximum half of the CPU cores,
            and up to 4 if the analysis is run in a regular computer.

        time_unit: string (default 'ps')
            String value use in the plotting axis when the results are display
        """
        self.fit_results = fit_results
        if bootstrap_result is None:
            self.bootstrap_result = self._generate_pandas_results_dataframe()
            self.data_simulated = None
        else:
            self.bootstrap_result = bootstrap_result
            self.data_simulated = bootstrap_result._type
        self.time_unit = time_unit
        self.confidence_interval = None
        self.datasets = None
        self.workers = workers
        self.fitter, self.params = self._get_original_fitter()

        # THE FOLLOWING ATTRIBUTES ARE FOR THE END OF THE PARALLEL COMPUTATION
        # self._cal_conf defines if to calculate the confidence interval at the
        # end of bootstrap for parallel computing. In that case
        # self._future_calculations is use to verify calculations are finished
        # self._names_futures store the conf-interval pandas DF columns names
        self._cal_conf = False
        self._future_calculations = 0
        self._names_futures = None

    def generate_datasets(self,
                          n_boots: int,
                          data_from='residues',
                          size=25,
                          return_data=False):
        """
        Method for generating simulated data sets from the data (shuffling), or
        from the residues. The las t approach shuffles the residues with the
        fitted model to generate the simulated data sets
        Parameters
        ----------
        n_boots: int
            Defines the number of samples that will be generated, we recommend
            to start with a low number, for example 5, fit this data and if
            everything is working simulate the rest and fit them.

        data_from: str (default residues)
            If "residues" data are simulated shuffling residues with the model.

            If "fitted_data" data are simulated from a random selection of the
            original fitted traces with replacement.

            If "full_data_matrix" data are simulated from a random selection of
            the original entire full data matrix with replacement.

            We recommend to either use 'residues' or 'full_data_matrix'.

            (Note that the globalfit, if run out of the experiment class, does
            not have access to the entire data matrix this data can be added
            with the fitResult.add_full_data_matrix(data, wavelength)

        size: int (default 25)
            Only important if the data_from is residues, defines the percentage
            of residues that will be shuffle. can be 10, 15, 20, 25, 33 or 50.
            We recommend to uses 25 or 33.

        return_data: bool (default False)
            If True the data set will be return
        """
        if self.data_simulated is not None:
            if data_from != self.data_simulated:
                msg = f'Cannot add new generated data from{data_from} ' \
                      f'to analysis of a {self.data_simulated} type'
                raise ExperimentException(msg)
        if data_from == 'residues':
            if hasattr(self.bootstrap_result, '_size'):
                size = self.bootstrap_result._size
            data = self._data_sets_from_residues(n_boots, size)
            self.bootstrap_result._type = 'residues'
            self.bootstrap_result._size = size
            self.data_simulated = 'residues'
        elif data_from == 'fitted_data':
            data = self._data_sets_from_data(n_boots, full_matrix=False)
            self.data_simulated = 'fitted_data'
            self.bootstrap_result._type = 'fitted_data'
        elif data_from == 'full_data_matrix':
            data = self._data_sets_from_data(n_boots, full_matrix=True)
            self.data_simulated = 'full_data_matrix'
            self.bootstrap_result._type = 'full_data_matrix'
        else:
            msg = 'data_from should be "residues" or "data"'
            raise ExperimentException(msg)
        self.datasets = data
        if return_data:
            return data

    def fit_bootstrap(self, cal_conf=True, parallel_computing=True):
        """
        Fit the simulated data sets with the same model used to obtain the
        fit_results passed to instatiate the obaject.
        Parameters
        ----------
        cal_conf: bool, (default True)
            If True the confidence intervals will be calculated after all fits
            have been completed
        parallel_computing: bool, (default False)
            If True the calculations will be run parallel using dask library

        """
        data_sets = self.datasets
        if data_sets is None:
            msg = 'Generate the data sets before'
            raise ExperimentException(msg)
        # extract parameters from the fit
        exp_no, type_fit, deconv, maxfev, tau_inf, use_jacobian = self._details()
        time_constraint = self.fit_results.details['time_constraint']
        weight = self.fit_results._weights
        method = self.fit_results.method
        if type(weight) == dict:
            apply_weight = weight['apply']
        else:
            apply_weight = False
        names = self._get_fit_params_names(type_fit, exp_no, deconv)
        variations = self._get_variations(names, exp_no)
        x = self.fit_results.x
        self._append_results_pandas_dataframe(self.bootstrap_result,
                                              self.fit_results, names)
        # fit all the generated data_sets
        if parallel_computing and self.workers != 1:
            self._cal_conf = cal_conf  # value recover by the add_done_callback
            self._names_futures = names
            self._future_calculations = 0
            calculations = []
            print("parallel computing")

            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers) as executor:

                for boot in range(data_sets.shape[2]):
                    data = data_sets[:, :, boot]
                    fitter = self.fitter(x, data, exp_no, self.params,
                                         deconv, tau_inf)
                    if apply_weight:
                        fitter.weights = weight

                    future_obj = executor.submit(fitter.global_fit, variations,
                                                 maxfev,
                                                 time_constraint,
                                                 apply_weight,
                                                 use_jacobian,
                                                 method)
                    fnc = self._append_results_pandas_dataframe_future
                    future_obj.add_done_callback(fnc)
                    calculations.append(future_obj)
                    print(f"parallel calculation {boot + 1} submitted")
        else:
            for boot in range(data_sets.shape[2]):
                data = data_sets[:, :, boot]
                fitter = self.fitter(x, data, exp_no, self.params,
                                     deconv, tau_inf)
                if apply_weight:
                    fitter.weights = weight
                results = fitter.global_fit(variations, maxfev=maxfev,
                                            time_constraint=time_constraint,
                                            apply_weights=apply_weight,
                                            use_jacobian=use_jacobian,
                                            method=method)
                self._append_results_pandas_dataframe(self.bootstrap_result,
                                                      results, names)
                print(f'Finished fit number: {boot + 1}')
            if cal_conf:
                conf = self.boot_conf_interval(data=self.bootstrap_result)
                self.confidence_interval = conf

    @staticmethod
    def boot_conf_interval(data):
        """
        Static method to calculate the confidence intervals from a dataFrame
        object obtained with the BootStrap class
        Parameters
        ----------
        data: dataFrame
            dataFrame containing the results of the fits to the simulated
            data sets
        Returns
        -------
        A pandas dataFrame with the calculated confidence intervals for:
        1-sigma, 2-sigma and 3-sigma
        """
        names = [i for i in data.keys() if 'final' in i]
        values = [0.27, 4.55, 31.7, -1, 68.27, 95.45, 99.73]
        table = pd.DataFrame(columns=['99.73%', '95.45%', '68.27%', '_BEST_',
                                      '68.27%', '95.45%', '99.73%'])
        print('conf calc')
        for i in names:
            array = data[i].values
            line = [np.percentile(array, val) - array[0] if val != -1
                    else array[0] for val in values]
            table.loc[i.split(' ')[0]] = line
        return table

    def plotBootStrapResults(self, param_1, param_2=None, kde=True):
        """
        Plot the bootstrap histogram of the decay times calculated
        If param_1 and param_2 are given a correlation plot with the
        histogram distributions is plot. If a single param is given only the
        histogram distribution is plot.

        Parameters
        ----------
        param_1: str or int
           name of the tau to be plotted;
            i.e.: for first decay time --> if string: tau1, if integer: 1

        param_2: str or int or None
            name of the tau to be plotted;
            i.e.: for third decay time --> if string: tau3, if integer: 3

        kde: bool (default True)
            Defines if the kernel density estimation is plotted
        """
        if type(param_1) == int:
            param_1 = f"tau{param_1}"
        if type(param_2) == int:
            param_2 = f"tau{param_2}"
        if param_2 is None:
            return self._plot_single_param(param_1, kde)
        else:
            return self._plot_double_param(param_1, param_2, kde)

    def _plot_single_param(self, param, kde=True):
        """
        Plot the histogram of a single param
        """
        fig, ax = plt.subplots(1, 1)
        bootstrap = self.bootstrap_result
        names = [i.split(' ')[0] for i in bootstrap.keys() if 'final' in i]
        stats = bootstrap.describe()
        stats_values = {}
        for name in names:
            stats_values[name + ' mean'] = \
                round(stats[name + ' final']['mean'], 4)
            stats_values[name + ' std'] = \
                round(stats[name + ' final']['std'], 4)
        if not kde:
            plt.ylabel('Counts')
            # plt.xlim(mini - maxi * 0.1, maxi + maxi * 0.1)
            stat = 'count'
        else:
            plt.ylabel('Density function')
            stat = 'density'
        ax = histplot(bootstrap[param + ' final'].values, kde=kde, stat=stat)
        
        plt.xlabel(f'Time ({self.time_unit})')
        maxi = bootstrap[param + ' final'].max()
        mini = bootstrap[param + ' final'].min()
        mean = bootstrap[param + ' final'].mean()
        dif_max = abs(maxi - mean)
        dif_min = abs(mini - mean)
        if dif_max > dif_min:
            pos = 1
        else:
            pos = 2
        mean = stats_values[param + ' mean']
        std = stats_values[param + ' std']
        tex = f'$\mu={mean}$ {self.time_unit}\n $\sigma={std}$ {self.time_unit}'
        texto = AnchoredText(s=tex, loc=pos)
        ax.add_artist(texto)
        return fig, ax

    def _plot_double_param(self, param_1, param_2, kde=True):
        """
        Plot a correlation plot between 2 decay times
        """
        if kde:
            label = 'Density'
            stat = 'density'
        else:
            label = 'Counts'
            stat = 'count'
        if 'k' not in param_1:  # in case exponential fit
            first_label = f'{self.time_unit}'
            second_label = f'{self.time_unit}'
        else:  # in case target fit
            first_label = f'1/{self.time_unit}'
            second_label = f'1/{self.time_unit}'

        grid_kw = {'height_ratios': [2, 5], 'width_ratios': [5, 2]}
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw=grid_kw)
        bootstrap = self.bootstrap_result
        alpha = self._get_alpha_for_plot_double_param(len(bootstrap))
        ax[0, 1].axis('off')
        ax[0, 0].set_xticklabels([])
        ax[1, 1].set_yticklabels([])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        # plot lateral histograms
        histplot(bootstrap[param_1 + ' final'].values, kde=kde, stat=stat,
                 ax=ax[0, 0])
        histplot(bootstrap, y=param_2 + ' final', kde=kde, stat=stat,
                 ax=ax[1, 1])

        # plot central area
        kdeplot(x=bootstrap[param_1 + ' final'].values, 
                y=bootstrap[param_2 + ' final'].values,
                ax=ax[1, 0], cmap='Spectral_r', shade=True)
        ax[1, 0].scatter(bootstrap[param_1 + ' final'],
                         bootstrap[param_2 + ' final'],
                         color='r', marker='+', alpha=alpha)

        # format axes
        ax[0, 0].set_ylabel(label)
        ax[1, 1].set_xlabel(label)
        ax[0, 0].set_xlabel('')
        ax[1, 1].set_ylabel('')
        ax[1, 0].set_xlabel(param_1.split(' ')[0] + f' ({first_label})')
        ax[1, 0].set_ylabel(param_2.split(' ')[0] + f' ({second_label})')

        return fig, ax

    def _data_sets_from_residues(self, n_boots, size=25):
        """
        Method for generating simulated data sets from  the residues.
        The function shuffles the residues with the fitted model to
        generate the simulated data sets
        Parameters
        ----------
        n_boots: int
            Defines the number of samples that will be generated, we recommend
            to start with a low number, for example 5, fit this data and if
            everything is working simulate the rest and fit them.
        size: int (default 25)
            Only important if the data_from is residues, defines the percentage
            of residues that will be shuffle. can be 10, 15, 20, 25, 33 or 50.
            We recommend to uses 25 or 33.
        """
        div = self._get_division_number(size)
        resultados = self.fit_results
        params = resultados.params
        data = resultados.data
        x = resultados.x
        deconv = resultados.details['deconv']
        result_explorer = ExploreResults(resultados)
        fittes = result_explorer.get_gloabl_fit_curve_results()
        residue_set_boot = resultados.data.copy()
        for boot in range(n_boots):
            residues = 0.0 * data
            for ii in range(len(residues[1])):
                if deconv:
                    residues[:, ii] = data[:, ii] - fittes[:, ii]
                else:
                    t0 = params['t0_1'].value
                    index = np.argmin([abs(i - t0) for i in x])
                    residues[index:, ii] = data[index:, ii] - fittes[:, ii]
                    data2 = 1.0 * data[:]
            for it in range(len(residues[1]) // div):
                value1 = np.random.randint(len(residues[1]))
                value2 = np.random.randint(len(residues[1]))
                residues[:, value1] = residues[:, value2]
            for da in range(len(residues[1])):
                if deconv:
                    data2[:, da] = fittes[:, da] + residues[:, da]
                else:
                    data2[index:, da] = fittes[:, da] + residues[index:, da]
            residue_set_boot = np.dstack((residue_set_boot, data2))
        residue_set_boot = residue_set_boot[:, :, 1:]
        return residue_set_boot

    def _data_sets_from_data(self, n_boots, full_matrix=True):
        """
        Method for generating simulated data sets from the data (shuffling).
        The method used numpy.random.choice  with replacement

        Parameters
        ----------
        n_boots: int
            Defines the number of samples that will be generated, we recommend
            to start with a low number, for example 5, fit this data and if
            everything is working simulate the rest and fit them.
        """
        # TODO full matrix
        number_traces = self.fit_results.data.shape[1]
        if full_matrix:
            if self.fit_results.full_matrix is not None:
                data = self.fit_results.full_matrix
            else:
                print("WARNING: the fit result does not contain the full data "
                      "matrix, the data sets have been generated from the "
                      "fitted traces. This is identical to the argument "
                      "'fitted_data'.")
        else:
            data = self.fit_results.data
        data_set_boot = data * 1.0
        for boot in range(n_boots):
            new_data = data * 0.0
            index = np.random.choice(np.linspace(0, len(data[1]) - 1,
                                                 len(data[1])), number_traces)
            for i, ii in enumerate(index):
                new_data[:, i] = data[:, int(ii)]
            data_set_boot = np.dstack((data_set_boot, new_data))
        return data_set_boot[:, :, 1:]

    def _get_variations(self, names, exp_no):
        """
        returns which parameter where varied in the original fit
        """
        variation = [self.fit_results.params[name].vary for name in names]
        return variation[-exp_no:]

    def _get_original_fitter(self):
        """
        returns which fitter was used in the original fit.
        Either GlobalFitTarget or GlobalFitExponential
        """
        exp_no, type_fit, deconv, maxfev, tau_inf, _ = self._details()
        initial_prams = deepcopy(self.fit_results.params)
        for i in initial_prams:
            initial_prams[i].value = initial_prams[i].init_value
        if type_fit == 'Exponential':
            fitter_obj = GlobalFitExponential
        elif type_fit == 'Target':
            fitter_obj = GlobalFitTarget
        else:
            msg = 'error in the defined type of fit'
            raise ExperimentException(msg)
        return fitter_obj, initial_prams

    def _details(self):
        """
        returns the detail of the original fit.
        """
        use_jacobian = self.fit_results.details['use_jacobian']
        exp_no = self.fit_results.details['exp_no']
        type_fit = self.fit_results.details['type']
        deconv = self.fit_results.details['deconv']
        maxfev = self.fit_results.details['maxfev']
        tau_inf = self.fit_results.details['tau_inf']
        return exp_no, type_fit, deconv, maxfev, tau_inf, use_jacobian

    def _get_division_number(self, size):
        """
        Returns the saffling number for residual data sets calculation
        according to the percentage of data to be shuffle.
        """
        if size not in [10, 15, 20, 25, 33, 50]:
            msg = 'Size should be either 10, 15, 20, 25, 33, this is ' \
                  'the values in percentage that will be randomly changed'
            raise ExperimentException(msg)
        elif size == 10:
            div = 10
        elif size == 15:
            div = 7
        elif size == 20:
            div = 5
        elif size == 25:
            div = 4
        elif size == 33:
            div = 3
        else:
            div = 2
        return div

    @staticmethod
    def _get_fit_params_names(type_fit, exp_no, deconv):
        """
        Returns the names of important parameters from the previous fit
        """
        if type_fit == 'Exponential':
            if deconv:
                names = ['t0_1', 'fwhm_1'] + ['tau%i_1' % (i + 1)
                                              for i in range(exp_no)]
            else:
                names = ['t0_1'] + ['tau%i_1' % (i + 1) for i in range(exp_no)]
        elif type_fit == 'Target':
            if deconv:
                names = ['t0_1', 'fwhm_1'] + ['k_%i%i' % (i + 1, i + 1)
                                              for i in range(exp_no)]
            else:
                names = ['t0_1'] + ['k_%i%i' % (i + 1, i + 1)
                                    for i in range(exp_no)]
        else:
            msg = 'Error defining type of fit'
            raise ExperimentException(msg)
        return names

    def _generate_pandas_results_dataframe(self):
        """
        Generates an empty Pandas dataFrame with important name values from the
        original fit, for time constant that where optimized
        """
        resultados = self.fit_results
        result_explorer = ExploreResults(resultados)
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, \
        type_fit, derivative_space = result_explorer._get_values(1)
        names = self._get_fit_params_names(type_fit, exp_no, deconv)
        initial_variations = [params[name].vary for name in names]
        # next 3 lines are for generate string names for the pandas data frame
        fit_names = [i[:-2] if 'k' not in i else i for ii, i in enumerate(names)
                     if initial_variations[ii]]
        col_name = [x for i in fit_names for x in
                    (i + ' initial', i + ' final')]
        detail_names = ['N° Iterations', 'red Xi^2', 'success']
        data_frame_names = col_name + detail_names
        residues_boostrap_result = pd.DataFrame(
            columns=data_frame_names)
        return residues_boostrap_result

    @staticmethod
    def _initial_final_values(params, names, only_vary=True):
        """
        Returns the initial and final values of a parameters object according to
        the names passed. If only vary is True, only those that were optimized
        are return
        """
        if only_vary:
            initial_values = [params[name].init_value for name in names
                              if params[name].vary]
            final_values = [params[name].value for name in names
                            if params[name].vary]
        else:
            initial_values = [params[name].init_value for name in names]
            final_values = [params[name].value for name in names]
        return initial_values, final_values

    def _append_results_pandas_dataframe_future(self, future):
        """
        Function for parallel computing
        """
        names = self._names_futures
        data_frame = self.bootstrap_result
        results = future.result()
        self._future_calculations += 1
        print(f"Finished calculation {self._future_calculations}")
        self._append_results_pandas_dataframe(data_frame, results, names)
        if self._future_calculations == self.datasets.shape[2]:
            print("All calculation  finished")
            if self._cal_conf:
                conf = self.boot_conf_interval(data=self.bootstrap_result)
                self._cal_conf = False
                self.confidence_interval = conf

    def _append_results_pandas_dataframe(self, data_frame, results, names):
        """
        Appends the results of the fit done to the simulated data sets
        to the pandas dataFrame containing the bootstrap results
        """
        params = results.params
        type_fit = results.details['type']
        key = data_frame.shape[0] + 1
        initial_values, final_values = self._initial_final_values(params,
                                                                  names)
        if type_fit == 'Target':
            # convert possible negative k value in positive
            # k values for a target fit may be negative because are
            # disappearance of a component
            final_values = [abs(ii) if i >= 1 else ii for i, ii
                            in enumerate(final_values)]
            initial_values = [abs(ii) if i >= 1 else ii for i, ii
                              in enumerate(initial_values)]
        fit_time_values = [x[i] for x in [(i, ii) for i, ii
                                          in zip(initial_values, final_values)]
                           for i in range(len(x))]
        fit_details = [float(results.nfev), float(results.redchi),
                       results.success]
        data_frame.loc[key] = fit_time_values + fit_details

    def _get_alpha_for_plot_double_param(self, number):
        if number > 250:
            alpha = 0.5
        elif number > 500:
            alpha = 0.25
        else:
            alpha = 1
        return alpha
