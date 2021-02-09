# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:47:20 2020

@author: 79344
"""
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtGui import  QIcon
from matplotlib.widgets import Slider, Button, RadioButtons
from ultrafast.MaptplotLibCursor import SnaptoCursor 
from pylab import pcolormesh


def correctGVDSellmeier(data,wavelenght,time): 
    GVD=ChripCorrection(data,wavelenght,time)
    correct_data=GVD.GVDFromGrapth()
    return correct_data

class ChripCorrection:
    def __init__(self,data,wavelenght,time):
        self.data=data
        self.x=time
        self.wavelenght=wavelenght
        self.corrected_data=None
        self._BK7_param={'b1':1.03961212,'b2':0.231792344,'b3':1.01046945,'c1':6.00069867e10-3,'c2':2.00179144e10-2,'c3':103.560653}
        self._SiO2_param={'b1':0.69616,'b2':0.4079426,'b3':0.8974794,'c1':4.67914826e10-3,'c2':1.35120631e10-2,'c3':97.9340025}
        self._CaF2_param={'b1':0.5675888,'b2':0.4710914,'b3':38.484723,'c1':0.050263605,'c2':0.1003909,'c3':34.649040}  
    
    def getCorrectData(self):
        if self.corrected_data == None:
            return 'data has not been corrected'
        else:
            return self.corrected_data
    
    def indexDifraction(self,x,b1,b2,b3,c1,c2,c3):
        n=(1+(b1*x**2/(x**2-c1**2))+(b2*x**2/(x**2-c2**2))+(b3*x**2/(x**2-c3**2)))**0.5    
        return n

    def fprime(self,x,b1,b2,b3,c1,c2,c3):
        return (b1*x**2/(-c1**2 + x**2) + b2*x**2/(-c2**2 + x**2) 
                + b3*x**2/(-c3**2 + x**2) + 1)**(-0.5)*(-1.0*b1*x**3/(-c1**2 + x**2)**2
                   + 1.0*b1*x/(-c1**2 + x**2) - 1.0*b2*x**3/(-c2**2 + x**2)**2
                   + 1.0*b2*x/(-c2**2 + x**2) - 1.0*b3*x**3/(-c3**2 + x**2)**2
                   + 1.0*b3*x/(-c3**2 + x**2))
    
    def dispersion(self,landa,element,excitation):
        b1,b2,b3,c1,c2,c3=element['b1'],element['b2'],element['b3'],element['c1'],element['c2'],element['c3']
        n_g=np.array([self.indexDifraction(i,b1,b2,b3,c1,c2,c3)-i*self.fprime(i,b1,b2,b3,c1,c2,c3) for i in landa])
        n_excitation=self.indexDifraction(excitation,b1,b2,b3,c1,c2,c3)-excitation*self.fprime(excitation,b1,b2,b3,c1,c2,c3)
        GVD=1/0.299792458*(n_excitation-n_g) #0.299792458 is the speed of light transform to correct units
        return GVD
        
    def GVD(self,CaF2=0,SiO2=0,BK7=0,offset=0):
        if self.dispersion_BK7 is 0:
             self.dispersion_BK7=self.dispersion(self.wavelength,self._BK7_param,self.excitation)
        if self.dispersion_Caf2 is 0:
            self.dispersion_Caf2=self.dispersion(self.wavelength,self._CaF2_param,self.excitation)
        if self.dispersion_SiO2 is 0:
            self.dispersion_SiO2=self.dispersion(self.wavelength,self._SiO2_param,self.excitation)
        print(CaF2,SiO2,BK7,offset)
        self.gvd=self.dispersion_BK7*BK7+self.dispersion_SiO2*SiO2+self.dispersion_Caf2*CaF2+offset
        self.CaF2,self.BK7,self.SiO2,self.GVD_offset=CaF2,BK7,SiO2,offset
        return self.gvd
    
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]
    
    def correctGVD(self,verified=False):
        result=self.gvd
        nx,ny=self.data.shape
        corrected_data=self.data.copy()
        valores=corrected_data.copy()*0.0
        for i in range(ny):
            new_time=[ii+result[i] for ii in self.x]
            for ii in range(len(new_time)):
                valor=new_time[ii]
                if valor < self.x[0]:
                    corrected_data[ii,i]=self.data[0,i]
                else:
                    idex,value=self.find_nearest(self.x, valor)
                    if value == valor:
                        corrected_data[ii,i]=self.data[idex,i]
                        valores[ii,i]=0
                    elif value < valor:
                        valores[ii,i]=-1
                        if idex==len(new_time)-1:
                            corrected_data[ii,i]=self.data[idex,i]
                        else:
                            sub=self.x[idex+1]
                            inf=self.x[idex]
                            w=abs((sub-valor)/(sub-inf))
                            corrected_data[ii,i]=w*self.data[idex,i]+(1-w)*self.data[idex+1,i]    
                    else:
                        valores[ii,i]=+1
                        inf=self.x[idex-1]
                        sub=self.x[idex]
                        w=abs((valor-sub)/(sub-inf))
                        corrected_data[ii,i]=w*self.data[idex-1,i]+(1-w)*self.data[idex,i]
        self.GVD_correction='in process'
#        fig, ax = plt.subplots(figsize=(6,6))
#        pcolormesh(self.wavelength,self.x[:46],pd.DataFrame(corrected_data).iloc[:46].values,cmap='RdYlBu')
        self.corrected_data=corrected_data
        if verified:
            self.verifiedGVD()
        else:
            return self.corrected_data
            
    def GVDFromPolynom(self,qt=None):
        self.gvd_Grapth=True
        self.figGVD=plt.figure(figsize=(7,6))
        result=np.array([self.x[0] for i in self.wavelength])
        self.l, = plt.plot(self.wavelength, result, lw=2, c='r')
        self.ax=self.figGVD.add_subplot(1,1,1)
        ylabel='Time ('+self.time_unit+')'
        plt.ylabel(ylabel,size=14)
        plt.xlabel('Wavelength (nm)',size=14)
        value=2
        if self.time_unit=='ns':
            value=value/1000
        self.index2ps=np.argmin([abs(i-value) for i in self.x])
        self.ax.pcolormesh(self.wavelength,self.x[:self.index2ps],pd.DataFrame(self.data).iloc[:self.index2ps].values,cmap='RdYlBu')
        plt.axis([self.wavelength[0],self.wavelength[-1], self.x[0], self.x[self.index2ps-1]])
        plt.subplots_adjust(bottom=0.15)
        self.cursor_pol=SnaptoCursor(self.ax,self.wavelength, self.x[:self.index2ps],draw='free',vertical_draw=False,single_line=False)
        self.figGVD.canvas.mpl_connect('button_press_event', self.cursor_pol.onClick)
        self.figGVD.canvas.mpl_connect('motion_notify_event', self.cursor_pol.mouseMove)
        self.figGVD.canvas.mpl_connect('axes_enter_event', self.cursor_pol.onEnterAxes)
        self.figGVD.canvas.mpl_connect('axes_leave_event', self.cursor_pol.onLeaveAxes)
        resetax2 = plt.axes([0.70, 0.025, 0.1, 0.04]) 
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(resetax, 'Calculate', color='tab:red', hovercolor='0.975')
        self.button2 = Button(resetax2, 'fit', color='tab:red', hovercolor='0.975')
        self.button.on_clicked(self.finalGVD)
        self.button2.on_clicked(self.fitPolGVD)
        if qt is not None:
            self.qt_path=qt
            thismanager = plt.get_current_fig_manager()
            thismanager.window.setWindowIcon(QIcon(qt))
        self.figGVD.show()
    
    def fitPolGVD(self,event):
        print('ok')
        point_pol_GVD=self.cursor_pol.datay
#        plt.close(self.fig)
        x=self.cursor_pol.datax
        wavelength=self.cursor_pol.x
        def optimize(params,x,y,order):
            return np.array([i**2 for i in range(len(x)+1,1,-1)])*(y-self.polynomi(params,x,order))
        params=lmfit.Parameters()
        params.add('c0',value=-5.7,max=0)
        params.add('c1',value=0.028,min=0)
        params.add('c2',value=-4.21E-5,max=0)
        params.add('c3',value=2.151E-8,min=0)
        wavelength=self.cursor_pol.x
        out = lmfit.minimize(optimize, params, args=(np.array(x),point_pol_GVD,3))
        self.gvd = self.polynomi(out.params,wavelength,3)
        self.l.set_ydata(self.gvd)
        self.figGVD.canvas.draw()        
        self.polynomGVD=True

#        print('top')
#        point_pol_GVD=experiment.cursor_pol.datay
#        def optimize(params,x,y,order):
#            print(order)
#            return (y-polynomi(params,x,order))
#        params=lmfit.Parameters()
#        params.add('c0',value=-5.7,max=0)
#        params.add('c1',value=0.028,min=0)
#        params.add('c2',value=-4.22E-5,max=0)
#        params.add('c3',value=2.153E-8,min=0)
#
#        poly=polynomi(params,wavelength,3)
#        experiment.l.set_ydata(poly)
#        experiment.figGVD.canvas.draw()
#        
#        x=experiment.cursor_pol.datax
#        wavelength=experiment.cursor_pol.x
#        out = lmfit.minimize(optimize, params, method='nelder',args=(np.array(x),point_pol_GVD,3))
#        poly = polynomi(out.params,wavelength,3)
#        experiment.l.set_ydata(poly)
#        experiment.figGVD.canvas.draw()
#        para=out.params
##        self.polynom_gvd=[out.params[key] for key in out.params]
#        experiment.polynomGVD=True
#        experiment.l.set_ydata(experiment.gvd)
#        experiment.figGVD.canvas.draw()
#        print('top')
#        weights=np.array([i**2 for i in range(len(x)+1,1,-1)])
#    
    def polynomi(self,params,x,order):
        pars=[params['c%i' %i].value for i in range(order+1)]
        return np.array([pars[i]*x**i for i in range(order+1)]).sum(axis=0)
    
    def GVDFromGrapth(self,qt=None):
        if self.data_before_first_selection is not None:
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
        self.gvd_Grapth=True
        self.GVD()
        self.figGVD=plt.figure(figsize=(7,6))
        #figGVD, ax = plt.subplots()
        self.figGVD.add_subplot()
        ylabel='Time ('+self.time_unit+')'
        plt.ylabel(ylabel,size=14)
        plt.xlabel('Wavelength (nm)',size=14)
        result=np.zeros(len(self.wavelength))
        self.l, = plt.plot(self.wavelength, result, lw=2, c='r')
        value=2
        if self.time_unit=='ns':
            value=value/1000
        self.index2ps=np.argmin([abs(i-value) for i in self.x])
        pcolormesh(self.wavelength,self.x[:self.index2ps],pd.DataFrame(self.data).iloc[:self.index2ps].values,cmap='RdYlBu')
        axcolor = 'lightgoldenrodyellow'
        plt.axis([self.wavelength[0],self.wavelength[-1], self.x[0], self.x[self.index2ps-1]])
        axamp = plt.axes([0.25, 0.01, 0.50, 0.02],facecolor=axcolor)
        axfreq = plt.axes([0.25, 0.055, 0.5, 0.02], facecolor=axcolor)
        plt.subplots_adjust(bottom=0.25)
        axofset=plt.axes([0.25, 0.135, 0.5, 0.02], facecolor=axcolor)
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
        self.button = Button(resetax, 'Calculate', color='tab:red', hovercolor='0.975')
        self.button.on_clicked(self.finalGVD)
        if qt is not None:
            self.qt_path=qt
            thismanager = plt.get_current_fig_manager()
            thismanager.window.setWindowIcon(QIcon(qt))
        self.figGVD.show()
    
    def updateGVD(self,val):
        # amp is the current value of the slider
        ofset=self.sofset.val 
        sio2=self.sfreq.val
        caf2 = self.samp.val
        bk=self.sbk7.val
        # update curve
        self.l.set_ydata(bk*self.dispersion_BK7+caf2*self.dispersion_Caf2+sio2*self.dispersion_SiO2+ofset)
        # redraw canvas while idle
        self.figGVD.canvas.draw_idle()
     
    def finalGVD(self,event):
        self.polynomGVD=False
        offset=self.sofset.val
        SiO2=self.sfreq.val
        CaF2=self.samp.val
        BK=self.sbk7.val
        self.GVD(CaF2=CaF2,SiO2=SiO2,BK7=BK,offset=offset)
        self.correctGVD(verified=True)

    def radioVerifiedGVD(self,label):
        radiodict={'True':True,'False':False}
        self.radio1=radiodict[label]
    
    def buttonVerifiedGVD(self,label):
        if self.radio1:
           self.GVD_correction=True
           print('Data has been corrected from GVD')
           plt.close(self.fig)
           plt.close()
           plt.close(self.figGVD)
           return self.corrected_data
        else:
           self.corrected_data=None
           if self.polynomGVD:
               plt.close(self.fig) 
               plt.close(self.figGVD)
               self.GVD_correction=False 
           else:
               self.GVD_correction=False 
               plt.close(self.fig) 
               plt.close(self.figGVD)
               plt.close()
               if self.gvd_Grapth:
                   self.GVDFromGrapth()
               print('Data has NOT been corrected from GVD')
    
    def verifiedGVD(self):
        assert self.GVD_correction=='in process', 'Please first try to correct the GVD'
        axcolor = 'lightgoldenrodyellow'
        self.fig=plt.figure(figsize=(12,6))
        ax0=self.fig.add_subplot(1,2,1)
        ax1=self.fig.add_subplot(1,2,2)
        values=[i for i in range(len(self.wavelength))[::round(len(self.wavelength)/11)]]
        ax0.pcolormesh(self.wavelength,self.x[:self.index2ps],pd.DataFrame(self.corrected_data).iloc[:self.index2ps].values,cmap='RdYlBu')
        #ax1 = self.fig.add_subplot(1,2,2)
#        self.corrected_data
        xlabel='Time ('+self.time_unit+')'
        for i in values[1:]:
            ax1.plot(self.x,self.corrected_data[:,i])
        plt.subplots_adjust(bottom=0.3)
        ax1.axvline(-0.5,linewidth=1,linestyle='--', color='k',alpha=0.4)
        ax1.axvline(0,linewidth=1,linestyle='--', color='k',alpha=0.4)
        ax1.axvline(0.5,linewidth=1,linestyle='--', color='k',alpha=0.4)
        ax1.axvline(1,linewidth=1,linestyle='--', color='k',alpha=0.4)
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button = Button(resetax, 'Apply', color='tab:red', hovercolor='0.975')
        rax = plt.axes([0.70, 0.025, 0.12, 0.12], facecolor=axcolor)
        ax1.axhline(linewidth=1,linestyle='--', color='k')
        ax1.ticklabel_format(style='sci',axis='y')
        #f.tight_layout()
        ax1.set_ylabel('$\Delta$A',size=14)
        ax1.set_xlabel(xlabel,size=14)
        ax1.minorticks_on()
        ax0.set_ylabel(xlabel,size=14)
        ax0.set_xlabel('Wavelength (nm)',size=14)
        self.radio = RadioButtons(rax, ('True', 'False'), active=0)
        self.radio1=self.radio.value_selected
        self.radio.on_clicked(self.radioVerifiedGVD)
        self.button.on_clicked(self.buttonVerifiedGVD)
        if self.qt_path is not None:
            thismanager = plt.get_current_fig_manager()
            thismanager.window.setWindowIcon(QIcon(self.qt_path))
        self.fig.show()