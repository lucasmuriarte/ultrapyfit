import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtGui import QIcon
from matplotlib.widgets import Slider, Button, RadioButtons
from pylab import pcolormesh


class Dispersions:
    BK7 = {'b1': 1.03961212, 'b2': 0.231792344,
           'b3': 1.01046945, 'c1': 6.00069867e10 - 3,
           'c2': 2.00179144e10 - 2, 'c3': 103.560653}
    SiO2 = {'b1': 0.69616, 'b2': 0.4079426, 'b3': 0.8974794,
            'c1': 4.67914826e10 - 3, 'c2': 1.35120631e10 - 2,
            'c3': 97.9340025}
    CaF2 = {'b1': 0.5675888, 'b2': 0.4710914, 'b3': 38.484723,
            'c1': 0.050263605, 'c2': 0.1003909, 'c3': 34.649040}

    def indexDifraction(self, x, b1, b2, b3, c1, c2, c3):
        n = (1 + (b1 * x ** 2 / (x ** 2 - c1 ** 2)) + (
                b2 * x ** 2 / (x ** 2 - c2 ** 2)) + (
                     b3 * x ** 2 / (x ** 2 - c3 ** 2))) ** 0.5
        return n

    def fprime(self, x, b1, b2, b3, c1, c2, c3):
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

    def dispersion(self, landa, element_name, excitation):
        element = getattr(self, element_name)
        b1, b2, b3, c1, c2, c3 = element['b1'], element['b2'], element['b3'], \
                                 element['c1'], element['c2'], element['c3']
        n_g = np.array([self.indexDifraction(i, b1, b2, b3, c1, c2,
                                             c3) - i * self.fprime(i, b1, b2,
                                                                   b3, c1, c2,
                                                                   c3) for i in
                        landa])
        n_excitation = self.indexDifraction(excitation, b1, b2, b3, c1, c2,
                                            c3) - excitation * self.fprime(
            excitation, b1, b2, b3, c1, c2, c3)
        GVD = 1 / 0.299792458 * (
                n_excitation - n_g)  # 0.299792458 is the speed of light transform to correct units
        return GVD


class ChripCorrection:
    def __init__(self, data, wavelength, time):
        self.data = data
        self.x = time
        self.wavelength = wavelength
        self.corrected_data = None
        self.gvd = None
        self.GVD_correction = False

    def correctGVD(self, verified=False):
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
                    idex, value = self.find_nearest(self.x, valor)
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
        self.GVD_correction = 'in process'
        #        fig, ax = plt.subplots(figsize=(6,6))
        #        pcolormesh(self.wavelength,self.x[:46],pd.DataFrame(corrected_data).iloc[:46].values,cmap='RdYlBu')
        self.corrected_data = corrected_data
        if verified:
            self.verifiedGVD()
        else:
            return self.corrected_data

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def radioVerifiedGVD(self, label):
        radiodict = {'True': True, 'False': False}
        self.radio1 = radiodict[label]

    def buttonVerifiedGVD(self, label):
        if self.radio1:
            self.GVD_correction = True
            print('Data has been corrected from GVD')
            plt.close(self.fig)
            plt.close()
            return self.corrected_data
        else:
            self.corrected_data = None
            self.GVD_correction = False
            plt.close(self.fig)
            plt.close()
            print('Data has NOT been corrected from GVD')

    def verifiedGVD(self):
        assert self.GVD_correction == 'in process', 'Please first try to correct the GVD'
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
        self.radio.on_clicked(self.radioVerifiedGVD)
        self.button.on_clicked(self.buttonVerifiedGVD)
        # if self.qt_path is not None:
        #    thismanager = plt.get_current_fig_manager()
        #    thismanager.window.setWindowIcon(QIcon(self.qt_path))
        self.fig.show()

class EstimationGVD:
    def __init__(self, data, wavelength, time):
        self.data = data
        self.x = time
        self.wavelength = wavelength
        self.corrected_data = None
        self.gvd = None

    def estimate_GVD(self):
        pass

    def estimate_GVD_from_grath(self):
        pass


class EstimationGVDSellmeier(EstimationGVD):
    def __init__(self, data, wavelength, time, excitation):
        super().__init__(data, wavelength, time)
        self.CaF2 = 0
        self.BK7 = 0
        self.SiO2 = 0
        self.GVD_offset = 0
        self.gvd = 0
        self.dispersion_BK7 = 0
        self.dispersion_Caf2 = 0
        self.dispersion_SiO2 = 0
        self._calculate_dispersions()
        self.excitation = excitation

    def estimate_GVD(self, CaF2=0, SiO2=0, BK7=0, offset=0):
        # print(CaF2, SiO2, BK7, offset)
        self.gvd = self.dispersion_BK7 * BK7 + self.dispersion_SiO2 * SiO2 + \
                   self.dispersion_Caf2 * CaF2 + offset
        self.CaF2, self.BK7, self.SiO2, self.GVD_offset = CaF2, BK7, SiO2, offset

    def estimate_GVD_from_grath(self, qt=None):
        self.figGVD = plt.figure(figsize=(7, 6))
        # figGVD, ax = plt.subplots()
        self.figGVD.add_subplot()
        ylabel = 'Time (ps)'
        plt.ylabel(ylabel, size=14)
        plt.xlabel('Wavelength (nm)', size=14)
        result = np.zeros(len(self.wavelength))
        self.l, = plt.plot(self.wavelength, result, lw=2, c='r')
        value = 2
        self.index2ps = np.argmin([abs(i - value) for i in self.x])
        pcolormesh(self.wavelength, self.x[:self.index2ps],
                   pd.DataFrame(self.data).iloc[:self.index2ps].values,
                   cmap='RdYlBu')
        axcolor = 'lightgoldenrodyellow'
        plt.axis([self.wavelength[0], self.wavelength[-1], self.x[0],
                  self.x[self.index2ps - 1]])
        axamp = plt.axes([0.25, 0.01, 0.50, 0.02], facecolor=axcolor)
        axfreq = plt.axes([0.25, 0.055, 0.5, 0.02], facecolor=axcolor)
        plt.subplots_adjust(bottom=0.25)
        axofset = plt.axes([0.25, 0.135, 0.5, 0.02], facecolor=axcolor)
        # Slider
        axbk7 = plt.axes([0.25, 0.095, 0.5, 0.02], facecolor=axcolor)
        self.sbk7 = Slider(axbk7, 'BK72', 0, 10, valinit=0, color='orange')
        self.samp = Slider(axamp, 'CaF2', 0, 10, valinit=0, color='g')
        self.sfreq = Slider(axfreq, 'SiO2', 0, 10, valinit=0, color='b')
        self.sofset = Slider(axofset, 'Offset', -2, 2, valinit=0, color='r')
        self.sbk7.on_changed(self.updateGVD)
        # call update function on slider value change
        self.samp.on_changed(self.updateGVD)
        self.sofset.on_changed(self.updateGVD)
        self.sfreq.on_changed(self.updateGVD)
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(resetax, 'Calculate', color='tab:red',
                             hovercolor='0.975')
        self.button.on_clicked(self.finalGVD)
        if qt is not None:
            self.qt_path = qt
            thismanager = plt.get_current_fig_manager()
            thismanager.window.setWindowIcon(QIcon(qt))
        self.figGVD.show()

    def updateGVD(self, val):
        # amp is the current value of the slider
        ofset = self.sofset.val
        sio2 = self.sfreq.val
        caf2 = self.samp.val
        bk = self.sbk7.val
        # update curve
        self.l.set_ydata(bk * self.dispersion_BK7 +
                         caf2 * self.dispersion_Caf2 +
                         sio2 * self.dispersion_SiO2 + ofset)
        # redraw canvas while idle
        self.figGVD.canvas.draw_idle()

    def finalGVD(self, event):
        self.polynomGVD = False
        offset = self.sofset.val
        SiO2 = self.sfreq.val
        CaF2 = self.samp.val
        BK = self.sbk7.val
        self.estimate_GVD(CaF2=CaF2, SiO2=SiO2, BK7=BK, offset=offset)

    def _calculate_dispersions(self):
        self.dispersion_BK7 = Dispersions.dispersion(self.wavelength, 'BK7',
                                                     self.excitation)
        self.dispersion_Caf2 = Dispersions.dispersion(self.wavelength, 'CaF2',
                                                      self.excitation)
        self.dispersion_SiO2 = Dispersions.dispersion(self.wavelength, 'SiO2',
                                                      self.excitation)