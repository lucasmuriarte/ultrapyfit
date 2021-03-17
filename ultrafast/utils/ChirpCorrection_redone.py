"""
Created on Mon Mar 15 21:40:16 2021

@author: lucas
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtGui import QIcon
from matplotlib.widgets import Slider, Button, RadioButtons
from pylab import pcolormesh
from ultrafast.utils.Preprocessing import ExperimentException
from ultrafast.graphics.MaptplotLibCursor import SnaptoCursor
import lmfit


class Dispersion:
    """
    Class that calculate the dispersion of different dispersive medium using
    the Sellmeier equation. It only contain static methods.
    The default dispersive media are BK7, SiO2 and CaF2
    """
    BK7 = {'b1': 1.03961212, 'b2': 0.231792344, 'b3': 1.01046945,
           'c1': 6.00069867e10 - 3, 'c2': 2.00179144e10 - 2, 'c3': 103.560653}
    SiO2 = {'b1': 0.69616, 'b2': 0.4079426, 'b3': 0.8974794,
            'c1': 4.67914826e10 - 3, 'c2': 1.35120631e10 - 2, 'c3': 97.9340025}
    CaF2 = {'b1': 0.5675888, 'b2': 0.4710914, 'b3': 38.484723,
            'c1': 0.050263605, 'c2': 0.1003909, 'c3': 34.649040}

    @staticmethod
    def indexDifraction(x, b1, b2, b3, c1, c2, c3):
        """
        Returns the index diffraction of a dispersive media at every wavelength
        pass (x). See https://en.wikipedia.org/wiki/Sellmeier_equation

        Parameters
        ----------
            x: int or float
                value of wavelength at which the dispersion is calculated
            b1, b2, b3, c1, c2, c3: float
                Values for the Sellmeier equation of a certain dispersive medium
        """
        n = (1 + (b1 * x ** 2 / (x ** 2 - c1 ** 2)) + (
                b2 * x ** 2 / (x ** 2 - c2 ** 2)) + (
                     b3 * x ** 2 / (x ** 2 - c3 ** 2))) ** 0.5
        return n

    @staticmethod
    def fprime(x, b1, b2, b3, c1, c2, c3):
        """
        Returns the index diffraction of a dispersive media at every wavelength
        pass (x).

        Parameters
        ----------
            x: int or float
                value of wavelength at which the dispersion is calculated
            x:
                Wavelength vector to calculate the   dispersion
            b1, b2, b3, c1, c2, c3:
                Values for the Sellmeier equation of a certain dispersive medium
        """
        return (b1 * x ** 2 / (-c1 ** 2 + x ** 2) + b2 * x ** 2 / (
                -c2 ** 2 + x ** 2)
                + b3 * x ** 2 / (-c3 ** 2 + x ** 2) + 1) ** (-0.5) * (
                       -1.0 * b1 * x ** 3 / (-c1 ** 2 + x ** 2) ** 2
                       + 1.0 * b1 * x / (
                               -c1 ** 2 + x ** 2) - 1.0 * b2 * x ** 3 / (
                               -c2 ** 2 + x ** 2) ** 2
                       + 1.0 * b2 * x / (
                               -c2 ** 2 + x ** 2) - 1.0 * b3 * x ** 3 / (
                               -c3 ** 2 + x ** 2) ** 2
                       + 1.0 * b3 * x / (-c3 ** 2 + x ** 2))

    @staticmethod
    def dispersion(wavelength, element_name, excitation):
        """
        Returns the index diffraction of a dispersive media at every wavelength
        pass (x). See https://en.wikipedia.org/wiki/Sellmeier_equation

        Parameters
        ----------
            wavelength: 1d numpy array or iterable
                Wavelength vector to calculate the   dispersion
            element_name: str
                a valid key for an attribute in the class, for example "BK7"
                "SiO2" and "CaF2"
            excitation: int or float
                reference value to calculate the dispersion normally should be
                the "Excitaion (pump)" wavelength used to excite the sample

        """

        element = getattr(Dispersion, element_name)
        b1, b2, b3, c1, c2, c3 = element['b1'], element['b2'], element['b3'], \
                                 element['c1'], element['c2'], element['c3']
        n_g = np.array([Dispersion.indexDifraction(i, b1, b2, b3, c1, c2, c3) -
                        i * Dispersion.fprime(i, b1, b2, b3, c1, c2, c3)
                        for i in wavelength])
        n_excitation = Dispersion.indexDifraction(excitation, b1, b2, b3, c1,
                                                  c2, c3) - excitation * \
                       Dispersion.fprime(excitation, b1, b2, b3, c1, c2, c3)
        # 0.299792458 is the speed of light transform to correct units
        GVD = 1 / 0.299792458 * (
                n_excitation - n_g) 
        return GVD


class ChripCorrection:
    """
    Class that corrects the group velocity dispersion (GVD) or chirp from
    a data set. The algorithm is based on the one proposed by Nakayama et al.
    (https://doi.org/10.1063/1.1148398), which used the data itself to correct
    the chrip.

    Attributes
    ----------
    x: 1darray
        x-vector, normally time vector

    data: 2darray
        Array containing the data, the number of rows should be equal to
        the len(x)

    wavelength: 1darray
            wavelength vector

    corrected_data: 2darray
        Array containing the data corrected, has the same shape as the
        original data

    gvd: 1darray
        contains the estimation of the dispersion and is used to correct
        the data

    GVD_corrected: bool
        True if the data has been corrected
    """
    def __init__(self, time, data, wavelength, gvd_estimation):
        """
        Constructor:

        Parameters
        ----------
        time: 1darray
            x-vector, normally time vector

        data: 2darray
            Array containing the data, the number of rows should be equal to
            the len(x)

        wavelength: 1darray
                wavelength vector

        gvd_estimation: 1darray
            contains the estimation of the dispersion and is used to
            correct the data
        """
        self.data = data
        self.x = time
        self.wavelength = wavelength
        self.corrected_data = None
        self.gvd = gvd_estimation
        self.GVD_corrected = False

    def correctGVD(self, verify=False):
        """
        Modified algorithm from Nakayama et al. to correct the chrip using the
        data itself.

        Parameter
        --------
        verify: bool (default False)
            If True a figure will pop out after the correction of the chirp,
            which allows to inspect the correction applied and decline
            or accept it
        """
        result = self.gvd
        nx, ny = self.data.shape
        corrected_data = self.data.copy()
        valores = corrected_data.copy() * 0.0
        for i in range(ny):
            new_time = [ii + result[i] for ii in self.x]
            for ii in range(len(new_time)):
                valor = new_time[ii]
                if valor < self.x[0]:
                    corrected_data[ii, i] = self.data[0, i]
                else:
                    idex, value = self._find_nearest(self.x, valor)
                    if value == valor:
                        corrected_data[ii, i] = self.data[idex, i]
                        valores[ii, i] = 0
                    elif value < valor:
                        valores[ii, i] = -1
                        if idex == len(new_time) - 1:
                            corrected_data[ii, i] = self.data[idex, i]
                        else:
                            sub = self.x[idex + 1]
                            inf = self.x[idex]
                            w = abs((sub - valor) / (sub - inf))
                            corrected_data[ii, i] = w * self.data[idex, i] + (
                                        1 - w) * self.data[idex + 1, i]
                    else:
                        valores[ii, i] = +1
                        inf = self.x[idex - 1]
                        sub = self.x[idex]
                        w = abs((valor - sub) / (sub - inf))
                        corrected_data[ii, i] = w * self.data[idex - 1, i] + (
                                    1 - w) * self.data[idex, i]
        self.GVD_corrected = 'in process'
#        fig, ax = plt.subplots(figsize=(6,6))
#        pcolormesh(self.wavelength,self.x[:46],pd.DataFrame(corrected_data).iloc[:46].values,cmap='RdYlBu')
        self.corrected_data = corrected_data
        if verify:
            self.verifiedGVD()
        else:
            return self.corrected_data

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def _radioVerifiedGVD(self, label):
        """
        Function used by verified GVD
        """
        radiodict = {'True': True, 'False': False}
        self.radio1 = radiodict[label]

    def _buttonVerifiedGVD(self, label):
        """
        Function used by verified GVD
        """
        if self.radio1:
            self.GVD_corrected = True
            print('Data has been corrected from GVD')
            plt.close(self.fig)
            plt.close()
            return self.corrected_data
        else:
            self.corrected_data = None
            self.GVD_corrected = False
            plt.close(self.fig)
            plt.close()
            print('Data has NOT been corrected from GVD')
            return None

    def verifiedGVD(self):
        """
        Function pops out a window that allows to check the correction of the
        chirp.
        """
        if self.GVD_corrected != 'in process':
            msg = 'Please first try to correct the GVD'
            raise ExperimentException(msg)
        axcolor = 'lightgoldenrodyellow'
        value = 2
        index2ps = np.argmin([abs(i - value) for i in self.x])
        self.fig = plt.figure(figsize=(12, 6))
        ax0 = self.fig.add_subplot(1, 2, 1)
        ax1 = self.fig.add_subplot(1, 2, 2)
        values = [i for i in range(len(self.wavelength))[
                             ::round(len(self.wavelength) / 11)]]
        ax0.pcolormesh(self.wavelength, self.x[:index2ps],
                       pd.DataFrame(self.corrected_data).iloc[
                       :index2ps].values, cmap='RdYlBu')
        # ax1 = self.fig.add_subplot(1,2,2)
        #        self.corrected_data
        xlabel = 'Time (ps)'
        for i in values[1:]:
            ax1.plot(self.x, self.corrected_data[:, i])
        plt.subplots_adjust(bottom=0.3)
        ax1.axvline(-0.5, linewidth=1, linestyle='--', color='k', alpha=0.4)
        ax1.axvline(0, linewidth=1, linestyle='--', color='k', alpha=0.4)
        ax1.axvline(0.5, linewidth=1, linestyle='--', color='k', alpha=0.4)
        ax1.axvline(1, linewidth=1, linestyle='--', color='k', alpha=0.4)
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(resetax, 'Apply', color='tab:red',
                             hovercolor='0.975')
        rax = plt.axes([0.70, 0.025, 0.12, 0.12], facecolor=axcolor)
        ax1.axhline(linewidth=1, linestyle='--', color='k')
        ax1.ticklabel_format(style='sci', axis='y')
        # f.tight_layout()
        ax1.set_ylabel('$\Delta$A', size=14)
        ax1.set_xlabel(xlabel, size=14)
        ax1.minorticks_on()
        ax0.set_ylabel(xlabel, size=14)
        ax0.set_xlabel('Wavelength (nm)', size=14)
        self.radio = RadioButtons(rax, ('True', 'False'), active=0)
        self.radio1 = self.radio.value_selected
        self.radio.on_clicked(self._radioVerifiedGVD)
        self.button.on_clicked(self._buttonVerifiedGVD)
        # if self.qt_path is not None:
        #    thismanager = plt.get_current_fig_manager()
        #    thismanager.window.setWindowIcon(QIcon(self.qt_path))
        self.fig.show()


class EstimationGVD:
    """
    Basic class to estimate and correct the GVD or chrip
    Attributes
    ----------
    time: 1darray
         normally the time vector

    data: 2darray
        Array containing the data, the number of rows should be equal to
        the len(x)

    wavelength: 1darray
        wavelength vector

    excitation: int or float (default None)
        Contains the excitation wavelength of the pump laser used

    corrected_data: 2darray
        Array containing the data corrected, has the same shape as the
        original data

    gvd: 1darray (default None)
        contains the estimation of the dispersion and is used to correct
        the data

    chirp_corrector: instance of ChripCorrection class
        use to correct the data
    """
    def __init__(self, time, data, wavelength, excitation=None):
        self.data = data
        self.corrected_data = None
        self.time = time
        self.wavelength = wavelength
        self.excitation = excitation
        self.corrected_done = False
        self.gvd = None
        self.chirp_corrector = ChripCorrection(self.time, self.data,
                                               self.wavelength, self.gvd)

    def estimate_GVD(self):
        """
        Method to estimate the GVD should be overwrite
        """
        pass

    def estimate_GVD_from_grath(self):
        """
        Method to estimate the GVD graphically should be overwrite
        """
        pass

    def correct_chrip(self, verify=True):
        """
        Modified algorithm from Nakayama et al. to correct the chirp using the
        data itself.

        Parameter
        --------
        verify: bool (default False)
            If True a figure will pop out after the correction of the chirp,
            which allows to inspect the correction applied and decline
            or accept it
        """
        if self.gvd is None:
            msg = "Estimate the Group velocity dispersion or chirp first"
            raise ExperimentException(msg)
        self.chirp_corrector.gvd = self.gvd
        self.corrected_data = self.chirp_corrector.correctGVD(verify=verify)
        corrected = False
        if self.corrected_data is not None:
            corrected = True
        self.corrected_done = corrected

    def get_corrected_data(self):
        return self.corrected_data


class EstimationGVDSellmeier(EstimationGVD):
    """
    Class that calculate the dispersion GVD or chirp using the Sellmeier
    equation. It only contain static methods.
    The default dispersive media are BK7, SiO2 and CaF2

    Attributes
    ----------
    BK7: int (Default 0)
        contains the millimeters of BK7

    SiO2: int (Default 0)
        contains the millimeters of SiO2

    CaF2: int (Default 0)
        contains the millimeters of CaF2

    GVD_offset: int (default 0)
        contains an offset time value, that can be used to correct the t0
        of the experimental data

    dispersion_BK7: 1darray
        contains dispersion curve of BK7

    dispersion_SiO2: 1darray
        contains dispersion curve of BK7

    dispersion_CaF2: 1darray
        contains dispersion curve of BK7

    """
    def __init__(self, time, data, wavelength, excitation):
        super().__init__(time, data, wavelength, excitation)
        self.CaF2 = 0
        self.BK7 = 0
        self.SiO2 = 0
        self.GVD_offset = 0
        self.dispersion_BK7 = 0
        self.dispersion_CaF2 = 0
        self.dispersion_SiO2 = 0
        self._calculate_dispersions()

    def estimate_GVD(self, CaF2=0, SiO2=0, BK7=0, offset=0, verify=True):
        """
        Estimates the total chirp as the sum of the dispersion introduced by all
        elements

        Parameters
        ----------
        BK7: int (Default 0)
            contains the millimeters of BK7

        SiO2: int (Default 0)
            contains the millimeters of SiO2

        CaF2: int (Default 0)
            contains the millimeters of CaF2

        offset: int (default 0)
            contains an offset time value, that can be used to correct the t0
            of the experimental data

        verify: bool (default True)
            If True, after correction a figure that allows to verify the
            correction will pop out.
        """
        # print(CaF2, SiO2, BK7, offset)
        self.gvd = self.dispersion_BK7 * BK7 + self.dispersion_SiO2 * SiO2 + \
                   self.dispersion_CaF2 * CaF2 + offset
        self.CaF2, self.BK7, self.SiO2, self.GVD_offset = CaF2, BK7, SiO2, offset
        self.correct_chrip(verify=verify)

    def estimate_GVD_from_grath(self, qt=None):
        """
        Allows to estimates the total chirp manually using sliders on the figure
        that pops out. The final chirp is the sum of the dispersion introduced
        by all elements.
        """
        self.figGVD = plt.figure(figsize=(7, 6))
        self.figGVD.add_subplot()
        plt.ylabel('Time (ps)', size=14)
        plt.xlabel('Wavelength (nm)', size=14)
        result = np.zeros(len(self.wavelength))  # original GVD value equal to 0
        self.l, = plt.plot(self.wavelength, result, lw=2, c='r')
        value = 2
        self.index2ps = np.argmin([abs(i - value) for i in self.time])
        pcolormesh(self.wavelength, self.time[:self.index2ps],
                   pd.DataFrame(self.data).iloc[:self.index2ps].values,
                   cmap='RdYlBu')
        plt.axis([self.wavelength[0], self.wavelength[-1], self.time[0],
                  self.time[self.index2ps - 1]])
        plt.subplots_adjust(bottom=0.25)
        # slider axis
        axcolor = 'lightgoldenrodyellow'
        axamp = plt.axes([0.25, 0.01, 0.50, 0.02], facecolor=axcolor)
        axfreq = plt.axes([0.25, 0.055, 0.5, 0.02], facecolor=axcolor)
        axofset = plt.axes([0.25, 0.135, 0.5, 0.02], facecolor=axcolor)
        axbk7 = plt.axes([0.25, 0.095, 0.5, 0.02], facecolor=axcolor)
        # sliders
        self.sbk7 = Slider(axbk7, 'BK72', 0, 10, valinit=0, color='orange')
        self.samp = Slider(axamp, 'CaF2', 0, 10, valinit=0, color='g')
        self.sfreq = Slider(axfreq, 'SiO2', 0, 10, valinit=0, color='b')
        self.sofset = Slider(axofset, 'Offset', -2, 2, valinit=0, color='r')
        # call update function on slider value change
        self.sbk7.on_changed(self._updateGVD)
        self.samp.on_changed(self._updateGVD)
        self.sofset.on_changed(self._updateGVD)
        self.sfreq.on_changed(self._updateGVD)
        # button
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(resetax, 'Calculate', color='tab:red',
                             hovercolor='0.975')
        self.button.on_clicked(self._finalGVD)
        if qt is not None:
            self.qt_path = qt
            thismanager = plt.get_current_fig_manager()
            thismanager.window.setWindowIcon(QIcon(qt))
        self.figGVD.show()

    def _updateGVD(self, val):
        """
        Function used by estimate_GVD_from_grath
        """
        # amp is the current value of the slider
        ofset = self.sofset.val
        sio2 = self.sfreq.val
        caf2 = self.samp.val
        bk = self.sbk7.val
        # update curve
        self.l.set_ydata(bk * self.dispersion_BK7 +
                         caf2 * self.dispersion_CaF2 +
                         sio2 * self.dispersion_SiO2 + ofset)
        # redraw canvas while idle
        self.figGVD.canvas.draw_idle()

    def _finalGVD(self, event):
        """
        Function used by estimate_GVD_from_grath
        """
        self.polynomGVD = False
        offset = self.sofset.val
        SiO2 = self.sfreq.val
        CaF2 = self.samp.val
        BK = self.sbk7.val
        self.estimate_GVD(CaF2=CaF2, SiO2=SiO2, BK7=BK, offset=offset,
                          verify=True)

    def _calculate_dispersions(self):
        """
        Calculate the dispersions curves of the different dispersive media
        """
        self.dispersion_BK7 = Dispersion.dispersion(self.wavelength, 'BK7',
                                                    self.excitation)
        self.dispersion_CaF2 = Dispersion.dispersion(self.wavelength, 'CaF2',
                                                     self.excitation)
        self.dispersion_SiO2 = Dispersion.dispersion(self.wavelength, 'SiO2',
                                                     self.excitation)


class EstimationGVDPolynom(EstimationGVD):
    """
    Class that allows to calculate the dispersion GVD or chirp fitting a
    polynomial to the data.
    """
    def __init__(self, time, data, wavelength):
        super().__init__(time, data, wavelength, None)

    def estimate_GVD_from_grath(self, qt=None):
        """
        Allows to estimates the total chirp manually by clicking on the figure
        that pops out. The final chirp is estimated fitting a polynomial to the
        selected points in the figure.
        """
        self.figGVD = plt.figure(figsize=(7, 6))
        self.l, = plt.plot(self.wavelength, self.wavelength * 0, lw=2, c='r')
        self.ax = self.figGVD.add_subplot(1, 1, 1)
        ylabel = 'Time (ps)'
        plt.ylabel(ylabel, size=14)
        plt.xlabel('Wavelength (nm)', size=14)
        value = 2
        self.index2ps = np.argmin([abs(i - value) for i in self.time])
        self.ax.pcolormesh(self.wavelength, self.time[:self.index2ps],
                           pd.DataFrame(self.data).iloc[
                           :self.index2ps].values, cmap='RdYlBu')
        plt.axis([self.wavelength[0], self.wavelength[-1], self.time[0],
                  self.time[self.index2ps - 1]])
        plt.subplots_adjust(bottom=0.15)
        self.cursor_pol = SnaptoCursor(self.ax, self.wavelength,
                                       self.time[:self.index2ps], draw='free',
                                       vertical_draw=False,
                                       single_line=False)
        self.figGVD.canvas.mpl_connect('button_press_event',
                                       self.cursor_pol.onClick)
        self.figGVD.canvas.mpl_connect('motion_notify_event',
                                       self.cursor_pol.mouseMove)
        self.figGVD.canvas.mpl_connect('axes_enter_event',
                                       self.cursor_pol.onEnterAxes)
        self.figGVD.canvas.mpl_connect('axes_leave_event',
                                       self.cursor_pol.onLeaveAxes)
        resetax3 = plt.axes([0.55, 0.025, 0.1, 0.04])
        resetax2 = plt.axes([0.70, 0.025, 0.1, 0.04])
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(resetax, 'Calculate', color='tab:red',
                             hovercolor='0.975')
        self.button2 = Button(resetax2, 'clear', color='tab:red',
                              hovercolor='0.975')
        self.button3 = Button(resetax3, 'fit', color='tab:red',
                              hovercolor='0.975')
        self.button.on_clicked(self.correct_chrip)
        self.button2.on_clicked(self._clear_fig)
        self.button3.on_clicked(self._fitPolGVD_Grapth)
        if qt is not None:
            self.qt_path = qt
            thismanager = plt.get_current_fig_manager()
            thismanager.window.setWindowIcon(QIcon(qt))
        self.figGVD.show()

    def _clear_fig(self, event):
        """
        Function used by estimate_GVD_from_grath, which clear the
        selected points
        """
        self.cursor_pol.clear()
        self.l.set_ydata(self.wavelength * 0)
        self.figGVD.canvas.draw()

    def _fitPolGVD_Grapth(self, event):
        """
        Function used by estimate_GVD_from_grath
        """
        y = self.cursor_pol.datay
        x = self.cursor_pol.datax
        self.fitPolGVD(x, y)
        self.l.set_ydata(self.gvd)
        self.figGVD.canvas.draw()

    def fitPolGVD(self, x, y):
        """
        fit a polynomial to the data point X and Y, automatically calculates the
        dispersion (gvd attribute) with the output of the polynomial fit and the
        wavelength attribute vector.
        """
        wavelength = self.wavelength
        # def optimize(params, x, y, order):
        #     return y - self.polynomi(params, x, order)

        # with weigths
        def optimize(params, x, y, order):
            return np.array([i**2 for i in range(len(x) + 1, 1, -1)]) * (
                        y - self.polynomi(params, x, order))

        params = lmfit.Parameters()
        params.add('c0', value=-5.7)
        params.add('c1', value=0.028)
        params.add('c2', value=-4.21E-4)
        params.add('c3', value=2.151E-7)
        params.add('c4', value=2.151E-9)
        out = lmfit.minimize(optimize, params,
                             args=(np.array(x), y, 4))
        self.gvd = self.polynomi(out.params, wavelength, 4)

    @staticmethod
    def polynomi(params, x, order):
        """
        Returns the values of a polynomial from a lmfit parameters object
        """
        pars = [params['c%i' % i].value for i in range(order + 1)]
        return np.array([pars[i] * x ** i for i in range(order + 1)]).sum(
            axis=0)
