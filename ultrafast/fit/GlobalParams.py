# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:48:51 2020

@author: 79344
"""
from lmfit import Parameters
import numpy as np
from ultrafast.fit.targetmodel import Model
from ultrafast.utils.Preprocessing import ExperimentException


class GlobExpParameters:
    """
    Class parameter generator global fitting

    attributes
    ----------
    taus: list
        contains the value of the decay times

    exp_no: int
        is the number of exponential and equal to the len of taus

    number_traces: int
        contains the number of data sets (traces) that should be globally fitted

    params: lmFit Parameters class
        contains the parameters for the fit
    """
    def __init__(self, number_traces: int, taus: list):
        """
        constructor function

        Parameters
        ----------
        number_traces: int
            number of data traces that will be fitted

        taus: list
            list containing the initial estimates for the exponential functions
        """
        
        self.taus = [taus] if type(taus) == float or type(taus) == int else taus
        self.exp_no = len(self.taus)
        self.number_traces = number_traces
        self.params = Parameters()
        
    def _generateParams(self, t0: float, vary_t0: bool):
        """ 
        generate the parameters for globally fitting the number of traces 
        to a sum of "exp_no" exponential decay
        """
        for iy in range(self.number_traces):
            self.params.add_many(
                ('y0_' + str(iy+1), 0, True, None, None, None, None),
                ('t0_' + str(iy+1), t0, vary_t0, None, None, None, None))
            
            for i in range(self.exp_no):
                # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                self.params.add_many(
                    ('pre_exp%i_' % (i+1) + str(iy+1), 0.1*10**(-i), True,
                     None, None, None, None),
                    ('tau%i_' % (i+1) + str(iy+1), self.taus[i], True,
                     0.00000001, None, None, None))

    def _add_deconvolution(self, fwhm: float, opt_fwhm: bool, tau_inf=1E12):
        """
        add the deconvolution parameters to a sum of "exp_no" exponential decay
        """
        for iy in range(self.number_traces):
            self.params.add_many(('fwhm_' + str(iy+1), fwhm, opt_fwhm, 0.000001, None, None, None))
            if tau_inf is not None:            
                self.params.add_many(('yinf_' + str(iy+1), 0.0, True, None, None, None, None))
            if iy > 0:
                self.params['fwhm_%i' % (iy+1)].expr = 'fwhm_1'

    def adjustParams(self, t0: float, vary_t0=True, fwhm=0.12, opt_fwhm=False,
                     GVD_corrected=True, tau_inf=1E12, y0=None):
        """
        function to initialize parameters for global fitting
        
        Parameters
        ----------
        t0: int or float
            the t0 for the fitting 
            
        vary_t0: bool (default True) 
            allows to optimize t0 when the sum of exponential is convolve with 
            a gaussian. If there is no deconvolution t0 is always fix to the 
            given value.
        
        fwhm: float or None (default 0.12)
            FWHM of the the laser pulse use in the experiment
            If None. the deconvolution parameters will not be added
        
        opt_fwhm: bool (default False)
            allows to optimized the FWHM. 
            Theoretically this should be measured externally and be fix
            (only applicable if fwhm is given)
        
        GVD_corrected: bool (default True) 
            If True all traces will have same t0. 
            If False t0 is independent of the trace
            
        tau_inf: int or float (default 1E12) 
            allows to add a constant decay value to the parameters.
            This modelled photoproducts formation with long decay times
            If None tau_inf is not added.
            (only applicable if fwhm is given)

        y0: int or float or list/1d-array (default None)
            If this parameter is pass y0 value will be a fixed parameter to the
            value passed. This affects fits with and without deconvolution. For
            a fit with deconvolution y0 is is added to negative delay offsets.
            For a fit without deconvolution y0 fit the offset of the exponential.
            If an array is pass this should have the length of the curves that
            want to be fitted, and for each curve the the y0 value would
            be different.
        """
        self._generateParams(t0, vary_t0)
        for iy in range(2, self.number_traces+1):
            self.params['t0_%i' % iy].expr = 't0_1'
            for i in range(self.exp_no):
                self.params['tau%i_' % (i+1) + str(iy)].expr = 'tau%i_1' % (i+1)
        if fwhm is not None:
            self._add_deconvolution(fwhm, opt_fwhm, tau_inf=tau_inf)
            if not GVD_corrected:
                for iy in range(2, self.number_traces+1):
                    self.params['t0_%i' % iy].expr = None
                    self.params['t0_%i' % iy].vary = True
        else:
            self.params['t0_1'].vary = False

        if y0 is not None:
            # this allow to pass a spectrum to consider the offset
            if not hasattr(y0, '__iter__'):
                # if is in case a single value is passes, create a list of y0
                # with same value
                y0 = [y0 for i in range(self.number_traces)]
            for iy in range(1, self.number_traces + 1):
                self.params['y0_%i' % iy].value = y0[iy-1]
                self.params['y0_%i' % iy].vary = False


class GlobalTargetParameters:
    """
    Class parameter generator global fitting

    attributes
    ----------
    model: model object
        contains model to be fitted

    exp_no: int
        is the number of exponential and equal to the len of taus

    number_traces: int
        contains the number of data sets (traces) that should be globally fitted

    params: lmFit Parameters class
        contains the parameters for the fit
    """
    def __init__(self, number_traces, model=None):
        """
        constructor function

        Parameters
        ----------
        number_traces: int
            number of data traces that will be fitted

        model: Model type object
            contains model to be fitted
        """
        self.params = None
        self.exp_no = None
        if model is not None:
            try:
                self.params_from_model(model)
            except Exception as e:
                model = None
                print("WARNING: Unable to generate parameters from model;" + e)
            finally:
                self.model = model
        else:
            self.model = None
        self.number_traces = number_traces

    def params_from_matrix(self, k_matrix: list, concentrations: list):
        """
        Build the k_matrix manually.

        Parameters
        ----------
        k_matrix: list of lists
            Contains all the information of k, rates. This parameter should be a
            list of list/tuples where every sub list should be of length and
            with the following form: [source, destination, rate constant, vary]
            if destination == source, parallel decay or terminal component.
                i.e.: [1, 3, 1/0.7, True] --> component 1 gives component 3 with
                a rate constant of 1/0.7, and this parameter can be optimized.

        concentrations: list
            a list containing the initial concentrations
        """
        self.params = Parameters()
        # find number of components/species
        sources = [i[0] for i in k_matrix]
        self.exp_no = np.max(sources)
        # verify length of concentrations agrees with the number of exponentials
        if len(concentrations) != self.exp_no:
            msg = "The number of concentrations and exponents do not match"
            raise ExperimentException(msg)
        else:
            total = sum(concentrations)
            for i in range(self.exp_no):
                self.params.add('c_%i' % (i+1), concentrations[i]/total, False)
        # generate params
        for i in range(self.exp_no):
            for ii in range(self.exp_no):
                self.params.add('k_%i%i' % (i+1, ii+1), 0, False)

        expresion = ["" for _ in range(self.exp_no)]
        for i in k_matrix:
            source = i[0]
            destination = i[1]
            rate = i[2]
            varied = i[3]
            if source != destination:
                self.params['k_%i%i' % (destination, source)].set(rate,
                                                                  vary=varied)
                expresion[source - 1] += '-k_%i%i' % (destination, source)
            else:
                # if destination == source, parallel decay or terminal component
                self.params['k_%i%i' % (destination, source)].set(-rate,
                                                                  vary=varied)

        for i in range(self.exp_no):
            if len(expresion[i]) > 0:
                self.params['k_%i%i' % (i + 1, i + 1)].set(expr=expresion[i])

    def params_from_model(self, model: Model):
        self.params = model.genParameters()
        self.exp_no = self.params['exp_no'].value

    def adjustParams(self, t0, vary_t0=True, fwhm=0.12, opt_fwhm=False,
                     GVD_corrected=True):
        """
        function to initialize parameters for global fitting

        Parameters
        ----------
        t0: int or float
            the t0 for the fitting

        vary_t0: bool (default True)
            allows to optimize t0 when the sum of exponential is convolve with
            a gaussian. If there is no deconvolution t0 is always fix to the
            given value.

        fwhm: float or None (default 0.12)
            FWHM of the the laser pulse use in the experiment
            If None. the deconvolution parameters will not be added

        opt_fwhm: bool (default False)
            allows to optimized the FWHM.
            Theoretically this should be measured externally and be fix
            (only applicable if fwhm is given)

        GVD_corrected: bool (default True)
            If True all traces will have same t0.
            If False t0 is independent of the trace
        """
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        if fwhm is not None:
            self._add_preexp_t0_y0(t0, vary_t0, GVD_corrected)
            self._add_deconvolution(fwhm, opt_fwhm)
        else:
            self._add_preexp_t0_y0(t0, False, True)

    def _add_deconvolution(self, fwhm, opt_fwhm):
        """
        add the deconvolution parameters to a sum of "exp_no" exponential decay
        """
        for iy in range(self.number_traces):
            self.params.add_many(('fwhm_' + str(iy+1), fwhm, opt_fwhm, 
                                  0.000001, None, None, None))
            if iy > 0:
                self.params['fwhm_%i' % (iy+1)].expr = 'fwhm_1'

    def _add_preexp_t0_y0(self, t0, vary_t0, GVD_corrected):
        """
        add t0 and pre_exponential values to the parameters from the model
        """
        for iy in range(self.number_traces):
            if iy > 0 and GVD_corrected:
                expres = 't0_1'
            else:
                expres = None
            self.params.add_many(
                ('y0_' + str(iy + 1), 0, True, None, None, None, None),
                # i fixed, may unfix later
                ('t0_' + str(iy + 1), t0, vary_t0, None, None, expres,
                 None))

            for i in range(self.exp_no):
                # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                self.params.add('pre_exp%i_' % (i + 1) + str(iy + 1),
                                0.1 * 10 ** (-i), True, None, None, None, None)

        for i in range(self.exp_no):
            if self.params['k_%i%i' % (i + 1, i + 1)].value != 0:
                self.params.add('tau%i_1' % (i + 1),
                                1/self.params['k_%i%i' % (i + 1, i + 1)].value,
                                vary=False)
            else:
                self.params.add('tau%i_1' % (i + 1),
                                np.inf,
                                vary=False)
