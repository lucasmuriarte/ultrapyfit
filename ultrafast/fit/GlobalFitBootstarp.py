import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from ultrafast.utils.Preprocessing import ExperimentException
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.fit.ExponentialFit import GlobalFitExponential
from ultrafast.fit.TargetFit import GlobalFitTargetModel
from seaborn import distplot
from matplotlib.offsetbox import AnchoredText


class BootStrap:
    def __init__(self, fit_results, bootstrap_result=None):
        self.fit_results = fit_results
        if bootstrap_result is None:
            self.bootstrap_result = self._generate_pandas_results_dataframe()
            self.data_simulated = None
        else:
            self.bootstrap_result = bootstrap_result
            self.data_simulated = bootstrap_result.bootstrap_type
        self.confidence_interval = None
        self.datas = None
        self.fitter, self.params =  self._get_original_fitter()

    def generate_data_sets(self, n_boots, size=25, data_from='residues', 
                           return_data=False):
        if self.data_simulated is not None:
            if data_from != self.data_simulated:
                msg = f'Cannot add new {data_from} analysis of a {self.data_simulated} type'
                raise ExperimentException(msg)
        if data_from == 'residues':               
            if hasattr(self.bootstrap_result, '_size'):
                size = self.bootstrap_result._size
            data = self._data_sets_from_residues(n_boots, size)
            self.bootstrap_type = 'residues'
            self.bootstrap_type._size = size
            self.data_simulated = 'residues'
        elif data_from == 'data':
            data = self._data_sets_from_residues(n_boots)
            self.data_simulated = 'data'
            data.bootstrap_type = 'data'
        else:
            msg ='data_from should be "residues" or "data"'
            raise ExperimentException(msg)
        self.datas = data
        if return_data:
            return data

    def fit_bootstrap(self, cal_conf=True):
        data_sets = self.datas
        if data_sets is None:
            msg = 'Generate the data sets before'
            raise ExperimentException(msg)
        exp_no, type_fit, deconv, maxfev, tau_inf = self._details()
        time_constraint = self.fit_results.details['time_constraint'] 
        weight = self.fit_results.weights
        names = self._get_fit_params_names(type_fit, exp_no, deconv)
        variations = self._get_variations(names, exp_no)
        x = self.fit_results.x
        self._append_results_pandas_dataframe(self.bootstrap_result,
                                              self.fit_results, names)
        for boot in range(data_sets.shape[2]):
            if type(weight) == dict:
                apply_weight = weight['apply'] 
            else:
                apply_weight = False
            data = data_sets[:, :, boot]
            fitter = self.fitter(x, data, exp_no, self.params, 
                                 deconv, tau_inf)
            if apply_weight:
                fitter.weights = weight
                
            reults = fitter.global_fit(variations, maxfev=maxfev,
                                      time_constraint=time_constraint,
                                      apply_weights=apply_weight)
            
            self._append_results_pandas_dataframe(self.bootstrap_result,
                                                  reults, names)

            print(f'the number of boots is: {boot}')
        if cal_conf:
            self.bootConfInterval(data=self.bootstrap_residues_result)

    @staticmethod
    def bootConfInterval(data):
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

    def plotBootStrapResults(self, fit_number, param, kde=False):
        fig, axes = plt.subplots(1, 1)
        bootsTrap = self.bootstrap_result
        names = [i.split(' ')[0] for i in bootsTrap.keys() if 'final' in i]
        stats = bootsTrap.describe()
        stats_values = {}
        for name in names:
            stats_values[name + ' mean'] = round(stats[name + ' final']['mean'], 4)
            stats_values[name + ' std'] = round(stats[name + ' final']['std'], 4)
        axes = distplot(bootsTrap[param + ' final'].values, 
                        rug=False, 
                        norm_hist=False, kde=kde,
                        hist_kws=dict(edgecolor="k",
                                      linewidth=2))
        plt.xlabel(f'Time ({self.time_unit})')
        maxi = bootsTrap[param + ' final'].max()
        mini = bootsTrap[param + ' final'].min()
        mean = bootsTrap[param + ' final'].mean()
        dif_max = abs(maxi - mean)
        dif_min = abs(mini - mean)
        if kde == False:
            plt.ylabel('Counts')
            plt.xlim(mini - maxi * 0.1, maxi + maxi * 0.1)
        else:
            plt.ylabel('Density function')
        if dif_max > dif_min:
            pos = 1
        else:
            pos = 2
        mean = stats_values[param + ' mean']
        std = stats_values[param + ' std']
        text = f'$\mu={mean}$ {self.time_unit}\n $\sigma={std}$ {self.time_unit}'
        texto = AnchoredText(s=text, loc=pos)
        axes.add_artist(texto)
        return fig, axes

    def _data_sets_from_residues(self, n_boots, size=25):
        div = self._get_division_number(size)
        resultados = self.fit_results
        data = resultados.data
        result_explorer = ExploreResults(resultados)
        fittes = result_explorer.results()
        residue_set_boot = resultados.data.copy()
        for boot in range(n_boots):
            residues = 0.0 * data
            for ii in range(len(residues[1])):
                residues[:, ii] = data[:, ii] - fittes[:, ii]
            for it in range(len(residues[1]) // div):
                value1 = np.random.randint(len(residues[1]))
                value2 = np.random.randint(len(residues[1]))
                residues[:, value1] = residues[:, value2]
            data2 = 0.0 * data[:]
            for da in range(len(residues[1])):
                data2[:, da] = fittes[:, da] + residues[:, da]
            residue_set_boot = np.dstack((residue_set_boot, data2))
        residue_set_boot = residue_set_boot[:, :, 1:]
        return residue_set_boot
    
    def _data_sets_from_data(self, n_boots):
        data = self.fit_results.data
        data_set_boot = data*1.0
        for boot in range(n_boots):
            new_data = data * 0.0
            index = np.random.choice(np.linspace(0, len(data[1]) - 1, 
                                                 len(data[1])), len(data[1]))
            for i, ii in enumerate(index):
                new_data[:, i] = self.data[:, int(ii)]
        return data_set_boot[:, :, 1:]
    
    def _get_variations(self, names, exp_no):
        variation = [self.fit_results.params[name].vary for name in names]
        return variation[-exp_no:]
    
    def _get_original_fitter(self):
        exp_no, type_fit, deconv, maxfev, tau_inf = self._details()
        initial_prams = deepcopy(self.fit_results.params)
        for i in initial_prams:
            initial_prams[i].value=initial_prams[i].init_value
        if type_fit == 'Exponential':
            fitter_obj = GlobalFitExponential     
        elif type_fit == 'Target':
            fitter_obj = GlobalFitTargetModel
        else:
            msg = 'error in the defined type of fit'
            raise ExperimentException(msg)
        return fitter_obj, initial_prams

    def _details(self):
        exp_no = self.fit_results.details['exp_no']
        type_fit = self.fit_results.details['type']
        deconv = self.fit_results.details['deconv']
        maxfev = self.fit_results.details['maxfev']
        tau_inf = self.fit_results.details['tau_inf']
        return exp_no, type_fit, deconv, maxfev, tau_inf

    def _get_division_number(self, size):
        if size not in [10, 15, 20, 25, 33]:
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
        return div
    
    @staticmethod
    def _get_fit_params_names(type_fit, exp_no, deconv):
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
        resultados = self.fit_results
        result_explorer = ExploreResults(resultados)
        x, data, wavelength, params, exp_no, deconv, tau_inf, svd_fit, \
            type_fit, derivative_space = result_explorer._get_values(1)
        names = self._get_fit_params_names(type_fit, exp_no, deconv)
        initial_variations = [params[name].vary for name in names]

        # next 3 lines are for generate string names for the pandas data frame
        fit_names = [i[:-2] if 'k' not in i else i for ii, i in enumerate(names) 
                     if initial_variations[ii]]
        col_name = [x for i in fit_names for x in (i+' initial', i+' final')]
        detail_names = ['N° Iterations', 'red Xi^2', 'success']
        data_frame_names = col_name + detail_names
        residues_boostrap_result = pd.DataFrame(
            columns=data_frame_names)
        return residues_boostrap_result

    @staticmethod
    def _initial_final_values(params, names, only_vary=True):
        if only_vary:
            initial_values = [params[name].init_value for name in names 
                              if params[name].vary]
            final_values = [params[name].value for name in names
                            if params[name].vary]
        else:
            initial_values = [params[name].init_value for name in names]
            final_values = [params[name].value for name in names]
        return initial_values, final_values

    def _append_results_pandas_dataframe(self, data_frame, results, names):
        params = results.params
        type_fit = results.details['type']
        key = data_frame.shape[0] + 1
        initial_values,  final_values = self._initial_final_values(params, 
                                                                    names)
        if type_fit == 'Target':
            # convert possible negative k value in positive
            # k values for a target fit may be negative because are disappearance of a component
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
