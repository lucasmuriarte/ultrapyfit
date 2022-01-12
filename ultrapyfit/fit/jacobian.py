import numpy as np
from ultrapyfit.fit.ModelCreator import ModelCreator
from scipy.special import erf


class Jacobian(ModelCreator):
    """
    'WARNING class under development'
    Class that has functions to calculate the Jacobian matrix analitically.
    It can be used as an alternative in the global fit classes or method of
    experiment, which by default resolve the Jacobian matrix numerically.
    If the Jabobian is used, it can speed-up the calculation.

    """
    def __init__(self, exp_no, time, tau_inf=1E+12):
        super().__init__(exp_no, time, tau_inf=tau_inf)


    @staticmethod
    def erfDerrivative(z):
        """
        calculate derrivative of the error function


        Parameters
        ----------
        z: float or array of floats
            function argument

        Returns
        ----------
        float or array of floats
        """

        return 2 * np.exp(-np.square(z)) / 1.7724538509055159

    def expNGaussDatasetJacobianByT0(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by time offset component (t0)
        in a equivalent way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """
        # i = lambda_i-1
        # tau_j is intentionally ignored
        t0 = params['t0_%i' % lambda_i].value
        fwhm = params['fwhm_%i' % lambda_i].value

        values = [[params['pre_exp%i_' % (ii + 1) + str(lambda_i)].value,
                   params['tau%i_' % (ii + 1) + str(lambda_i)].value]
                  for ii in range(self.exp_no)]
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (lambda_i)].value
            values.append([yinf, self.tau_inf])

        return sum(
            [0.5 * pre_exp * self.expGaussDerrivativeT0(self.x - t0,
                                                        tau,
                                                        fwhm / 2.35482)
             for pre_exp, tau in values])

    def expNGaussDatasetJacobianByY0(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by global offset component (y0)
        in a equivalent way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """

        return np.ones(self.x.shape[0])

    @staticmethod
    def expGaussDerrivativeT0(time, tau, sigma):
        """
        exponential derrivative by t0

        Parameters
        ----------
        time: array type
            the time vector

        tau: float or int
            decay associated time value

        sigma: float or int
            variance of the gaussian distribution

        Returns
        ----------
        1darray of size equal to time-vector
        """

        inv_tau = 1 / tau
        inv2_tau = inv_tau ** 2
        erf_part = 1 + erf((time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5))
        exp_part = np.exp(-inv_tau * time + sigma ** 2 * inv2_tau / 2)

        return exp_part * erf_part * inv_tau + exp_part * \
               Jacobian.erfDerrivative(
                   (time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5)) * \
               (-1 / (sigma * 2 ** 0.5))

    def expNGaussDatasetJacobianByY0(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by global offset component (y0)
        in a equivalent way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """

        return np.ones(self.x.shape[0])

    def expNGaussDatasetJacobianByYInf(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by pre_exp of inifinite component
        (yinf) in a equivalent way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """
        # i = lambda_i-1
        t0 = params['t0_%i' % lambda_i].value
        fwhm = params['fwhm_%i' % lambda_i].value

        tau = self.tau_inf

        time = self.x - t0
        sigma = fwhm / 2.35482

        inv_tau = 1 / tau
        inv2_tau = inv_tau ** 2
        erf_part = 1 + erf((time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5))
        exp_part = np.exp(-inv_tau * time + sigma ** 2 * inv2_tau / 2)

        return 0.5 * exp_part * erf_part

    def expNGaussDatasetJacobianByPreExp(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by pre_exp in a equivalent
        way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """
        # i = lambda_i-1
        t0 = params['t0_%i' % (lambda_i)].value
        fwhm = params['fwhm_%i' % (lambda_i)].value

        tau = params['tau%i_' % (tau_j) + str(lambda_i)].value

        time = self.x - t0
        sigma = fwhm / 2.35482

        inv_tau = 1 / tau
        inv2_tau = inv_tau ** 2
        erf_part = 1 + erf((time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5))
        exp_part = np.exp(-inv_tau * time + sigma ** 2 * inv2_tau / 2)

        return 0.5 * exp_part * erf_part

    @staticmethod
    def expGaussDerrivativeSigma(time, tau, sigma):
        """
        exponential derrivative by sigma

        Parameters
        ----------
        time: array type
            the time vector

        tau: float or int
            decay associated time value

        sigma: float or int
            variance of the gaussian distribution

        Returns
        ----------
        1darray of size equal to time-vector
        """

        inv_tau = 1 / tau
        inv2_tau = inv_tau ** 2
        erf_part = 1 + erf((time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5))
        exp_part = np.exp(-inv_tau * time + sigma ** 2 * inv2_tau / 2)

        return exp_part * erf_part * (sigma * inv2_tau) + exp_part * \
               Jacobian.erfDerrivative(
                   (time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5)) * \
               (-time / (sigma ** 2 * 2 ** 0.5) - inv_tau / (2 ** 0.5))

    def expNGaussDatasetJacobianBySigma(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by tau in a equivalent
        way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """
        # i = lambda_i-1
        # tau_j is intentionally ignored
        t0 = params['t0_%i' % lambda_i].value
        fwhm = params['fwhm_%i' % lambda_i].value

        values = [[params['pre_exp%i_' % (ii + 1) + str(lambda_i)].value,
                   params['tau%i_' % (ii + 1) + str(lambda_i)].value]
                  for ii in range(self.exp_no)]
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % lambda_i].value
            values.append([yinf, self.tau_inf])

        return sum(
            [0.5 * pre_exp * self.expGaussDerrivativeSigma(self.x - t0,
                                                           tau,
                                                           fwhm / 2.35482) / 2.35482
             for pre_exp, tau in values])

    def expNGaussDatasetJacobianByTau(self, params, lambda_i, tau_j):
        """
        calculate jacobian derrivative by tau in a equivalent
        way to the expNGaussDataset method.

        Parameters
        ----------
        params: GlobExpParameters  object
          object containing the parameters created for global
          fitting several decay traces

        lambda_i: int
            number corresponding to the specific trace
            it is from 1 to self.exp_no (like in param key)

        tau_j: int
            number corresponding to the specific tau number
            it is from 1 to self.exp_no (like in param key)

        Returns
        ----------
        1darray of size equal to time-vector
        """
        # i = lambda_i-1
        t0 = params['t0_%i' % lambda_i].value
        fwhm = params['fwhm_%i' % lambda_i].value

        pre_exp = params['pre_exp%i_%i' % (tau_j, lambda_i)].value
        tau = params['tau%i_%i' % (tau_j, lambda_i)].value

        time = self.x - t0
        sigma = fwhm / 2.35482

        inv_tau = 1 / tau
        inv2_tau = inv_tau ** 2
        inv3_tau = inv_tau ** 3
        erf_part = 1 + erf((time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5))
        exp_part = np.exp(-inv_tau * time + sigma ** 2 * inv2_tau / 2)

        tmp = exp_part * erf_part * (inv2_tau * time - sigma ** 2 * inv3_tau) + \
              exp_part * Jacobian.erfDerrivative(
            (time - sigma ** 2 * inv_tau) / (sigma * 2 ** 0.5)) * \
              (sigma ** 2 * inv2_tau) / (sigma * 2 ** 0.5)

        return 0.5 * pre_exp * tmp