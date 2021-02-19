# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:45:11 2019

@author: 79344
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class SnaptoCursor2(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()


class Calibration:
    """
    allow to calculate the correlation polynom (calibration curve) of order N between two data sets
    
    Attributes
    ----------
    ref_data: pandas Data Frame (defautlt: None)
        contain the datareference normally X-axis is in wavelength
    
    experimental_data: pandas Data Frame (defautlt: None)
        contain the experimental data to be calibrated normally X-axis is in pixel
    """
    def __init__(self,ref_data=None,experimental_data=None):
        if experimental_data is not None:
            self.data=experimental_data.reset_index(0,drop=True)
            self.data.columns = ['Pixel', 'Absorbance']
        
        if ref_data is not None:
            self.ref=ref_data.reset_index(0,drop=True)
            self.ref.columns = ['Wavelength', 'Absorbance']
        self.data_find={'min':-0.2,'width':1,'prominence':None,}
        self.ref_find={'min':-0.2,'width':2,'prominence':(0.01, None),}
        
    def plotDataRef(self,find_peaks=True):
        """
        make a subplot of the experimental_data and the reference_data 
        
        Parameters
        ----------
        find_peaks: bool (default True)
            if True find and plot the peaks of the experimental_data and reference_data,
            using the data_find  and ref_find attribute parameters 
            
        """
        X_data , y_data =self.data['Pixel'],self.data['Absorbance']
        X_ref , y_ref =self.ref['Wavelength'],self.ref['Absorbance']
        fig, ax= plt.subplots(nrows=1,ncols=2,figsize=(15,6))
        ax[0].plot(X_data , y_data)
        ax[0].set_xlabel('Pixel')
        ax[0].axhline(linewidth=1,linestyle='--', color='k')
        ax[1].plot(X_ref , y_ref)
        ax[1].set_xlabel('Wavelenght')
        ax[1].axhline(linewidth=1,linestyle='--', color='k')
        if find_peaks:
            X_data , y_data =self.data['Pixel'],self.data['Absorbance']
            data_peaks,data_properties=self.autoFindPeaks(X_data,y_data,height=self.data_find['min']\
                                ,width=self.data_find['width'],prominence=self.data_find['prminence'],plot=False)
            X_ref , y_ref =self.ref['Wavelength'],self.ref['Absorbance']
            ref_peaks,ref_properties=self.autoFindPeaks(X_ref,y_ref,height=self.ref_find['min'],\
                                width=self.ref_find['width'],prominence=self.ref_find['prminence'],plot=False)
            ax[0].plot(X_data[data_peaks], y_data[data_peaks], "x")
            ax[0].vlines(X_data[data_peaks], ymin=y_data[data_peaks] - data_properties["prominences"],ymax = y_data[data_peaks], color = "C1")
            ax[0].hlines(y=data_properties["width_heights"], xmin=data_properties["left_ips"],
                     xmax=data_properties["right_ips"], color = "C1")
            ax[0].set_xlabel('Pixel')
            ax[1].plot(X_ref[ref_peaks], y_ref[ref_peaks], "x")
            ax[1].vlines(X_ref[ref_peaks], ymin=y_ref[ref_peaks] - ref_properties["prominences"],ymax = y_ref[ref_peaks], color = "C1")
            ax[1].hlines(y=ref_properties["width_heights"], xmin=ref_properties["left_ips"],
                     xmax=ref_properties["right_ips"], color = "C1")
            ax[1].set_xlabel('Wavelenght')
        plt.show()
        
    def autoFindPeaks(self,X,y,height,width,prominence=None,plot=True):
        peaks, properties = find_peaks(y, height=height, width=2,prominence=prominence)
        X_distance=np.mean([abs(X[i]-X[ii]) for i,ii in enumerate(range(len(X)))])
        properties["left_ips"]=X[0]+properties["left_ips"]*X_distance
        properties["right_ips"]=X[0]+properties["right_ips"]*X_distance
        if plot:
            plt.figure()
            plt.plot(X,y)
            plt.plot(X[peaks], y[peaks], "x")
            plt.vlines(X[peaks], ymin=y[peaks] - properties["prominences"],ymax = y[peaks], color = "C1")
            plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                     xmax=properties["right_ips"], color = "C1")
            #plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
            #        xmax=properties["right_ips"], color = "C1")
            plt.show()
        return peaks, properties
    
    def doCalibrationFit(self,X1,X2,fit_order):
        """
        do a polynomial fit of order=fit_order between X1 and X2
        
        Parameters
        ----------
        X1 : array_like
            cordinate points of the experimental_data to be correlated with X2 (ref_data)
        X2 : array_like,
            cordinate points of the of the ref_data
            
        retunr:
            the polynom of order=fit_order and the R2 of the fit
        
        """
        polynom=np.poly1d(np.polyfit(X1,X2, fit_order))
        # fit values, and mean
        yhat = polynom(X1)                   
        ybar = np.sum(X2)/len(X2)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((X2 - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        R2 =ssreg / sstot
        return polynom,R2
    
    def calibrationFromGrapth(self, fit_order=1):
        X_data , y_data =self.data['Pixel'],self.data['Absorbance']
        X_ref , y_ref =self.ref['Wavelength'],self.ref['Absorbance']
        fig, ax= plt.subplots(figsize=(12,6))
        ax.plot(X_data , y_data)
        ax.set_xlabel('Pixel')
        ax.axhline(linewidth=1,linestyle='--', color='k')
        ax.set_xlim(X_data.values[0],X_data.values[-1])
        cursor = SnaptoCursor2(ax, X_data, y_data)
        cid =  plt.connect('motion_notify_event', cursor.mouse_move)
        data_points=plt.ginput(n=-1,timeout=-1,show_clicks=True)
        data_points=sorted([i[0] for i in data_points])
        plt.show()
        plt.close(fig)
        data_peaks=[(self.data['Pixel']-i).abs().sort_values().index[0] for i in data_points]
        pixels=X_data[data_peaks] 
        
        fig, ax= plt.subplots(figsize=(12,6))
        ax.plot(X_ref , y_ref)
        ax.set_xlabel('Wavelength')
        ax.axhline(linewidth=1,linestyle='--', color='k')
        ax.set_xlim(X_ref.values[0],X_ref.values[-1])
        cursor = SnaptoCursor2(ax, X_ref, y_ref)
        cid =  plt.connect('motion_notify_event', cursor.mouse_move)
        ref_points=plt.ginput(n=len(data_points),timeout=-1,show_clicks=True)
        ref_points=sorted([i[0] for i in ref_points])
        plt.show()
        plt.close(fig)
        ref_peaks=[(self.ref['Wavelength']-i).abs().sort_values().index[0] for i in ref_points]
        wavelength=X_ref[ref_peaks]
        
        self.calibration_curve,R2=self.doCalibrationFit(pixels,wavelength,fit_order)
        x=np.linspace(X_data.values[0],X_data.values[-1],len(X_data))
        lista=[round(self.calibration_curve[i],2) for i in range(fit_order+1)]
        legenda=[str(lista[0])]+[str(lista[i])+'X$^'+str(i)+'$' for i in range(1,fit_order+1)]
        string='+'.join(legenda)
        fig, ax= plt.subplots(nrows=1,ncols=3,figsize=(20,6))
        ax[0].plot(X_data , y_data)
        ax[0].plot(X_data[data_peaks], y_data[data_peaks], "x")
        ax[0].set_xlabel('Pixel')
        ax[1].plot(X_ref , y_ref)
        ax[1].plot(X_ref[ref_peaks], y_ref[ref_peaks], "x")
        ax[1].set_xlabel('Wavelength')
        ax[2].scatter(pixels,wavelength,color='red',label='Points')
        ax[2].plot(x,self.calibration_curve(x),label=rf'R$^2$ {round(R2,3)}')
        ax[2].set_title(f'polynom = {string}')
        ax[2].legend()
        ax[2].set_xlabel('Pixel')
        ax[2].set_ylabel('Wavelength')
        plt.show()
    
    def autoCalibrationWithSpectra(self, plot=True,fit_order=1):
        """
        Try to auto calibrate the data given the attributes ref_data and experimental_data 
        
        Parameters
        ----------
        
        plot: bool (default True)
            plot experimental_data and ref_data with the automatic founded peaks and a third subplot with the calibration curve found 
        
        fit_order: int or float the order for the fitting curve that correlates the experimental_data and ref_data
                    normally should be either 1 or 2
        """
        fit_order=round(fit_order)
        X_data , y_data =self.data['Pixel'],self.data['Absorbance']
        data_peaks,data_properties=self.autoFindPeaks(X_data,y_data,height=self.data_find['min'],\
                        width=self.data_find['width'],prominence=self.data_find['prminence'],plot=False)
        pixels=X_data[data_peaks]
        X_ref , y_ref =self.ref['Wavelength'],self.ref['Absorbance']
        ref_peaks,ref_properties=self.autoFindPeaks(X_ref,y_ref,height=self.ref_find['min'],\
                        width=self.ref_find['width'],prominence=self.ref_find['prminence'],plot=False)
        wavelength=X_ref[ref_peaks]
        assert len(pixels)==len(wavelength), ('The number of peaks found in the reference and the data are diferent')
        self.calibration_curve,R2=self.doCalibrationFit(pixels,wavelength,fit_order)
        
        x=np.linspace(X_data.values[0],X_data.values[-1],len(X_data))
        lista=[round(self.calibration_curve[i],2) for i in range(fit_order+1)]
        legend=[str(lista[0])]+[str(lista[i])+'X$^'+str(i)+'$' for i in range(1,fit_order+1)]
        string='+'.join(legend)
        if plot:
            fig, ax= plt.subplots(nrows=1,ncols=3,figsize=(20,6))
            ax[0].plot(X_data , y_data)
            ax[0].plot(X_data[data_peaks], y_data[data_peaks], "x")
            ax[0].vlines(X_data[data_peaks], ymin=y_data[data_peaks] - data_properties["prominences"],ymax = y_data[data_peaks], color = "C1")
            ax[0].hlines(y=data_properties["width_heights"], xmin=data_properties["left_ips"],
                     xmax=data_properties["right_ips"], color = "C1")
            ax[0].set_xlabel('Pixel')
            ax[1].plot(X_ref , y_ref)
            ax[1].plot(X_ref[ref_peaks], y_ref[ref_peaks], "x")
            ax[1].vlines(X_ref[ref_peaks], ymin=y_ref[ref_peaks] - ref_properties["prominences"],ymax = y_ref[ref_peaks], color = "C1")
            ax[1].hlines(y=ref_properties["width_heights"], xmin=ref_properties["left_ips"],
                     xmax=ref_properties["right_ips"], color = "C1")
            ax[1].set_xlabel('Wavelength')
            ax[2].scatter(pixels,wavelength,color='red',label='Points')
            ax[2].plot(x,self.calibration_curve(x),label=rf'R$^2$ {round(R2,3)}')
            ax[2].set_title(f'polynom = {string}')
            ax[2].legend()
            ax[2].set_xlabel('Pixel')
            ax[2].set_ylabel('Wavelength')
            plt.show()
        return self.calibration_curve
    
    def cutRef(self,rango):
        self.ref=self.ref[(self.ref['Wavelength']>rango[0]) & (self.ref['Wavelength']<rango[1])].reset_index(0,drop=True)
        
    def cutData(self,rango):
        self.data=self.data[(self.data['Wavelength']>rango[0]) & (self.data['Wavelength']<rango[1])].reset_index(0,drop=True)
    
    def manualCalibration(self,lists_pixels,lists_wavelengths,fit_order):
        self.calibration_curve,R2=self.doCalibrationFit(lists_pixels,lists_wavelengths,fit_order)
        
    def importExperimentalData(self,path, plot=False,header=None,sep=','):
        self.data=pd.read_csv(path,sep=sep,header=header,names=['Pixel','Absorbance'])
        if plot:
            fig=plt.figure()
            ax=plt.plot(self.data['Pixel'],self.data['Absorbance'])
            plt.xlabel('Pixel')
            plt.ylabel('Absorbance')
            return fig,ax
        
    def importReferenceData(self,path,rango=None,plot=False,header=None,sep=','):
        self.ref=pd.read_csv(path,sep=sep,header=header,names=['Wavelength','Absorbance'])
        if rango!=None:
            self.ref=self.ref[(self.ref['Wavelength']>rango[0]) & (self.ref['Wavelength']<rango[1])].reset_index(0,drop=True)
        if plot:
            fig=plt.figure()
            ax=plt.plot(self.ref['Wavelength'],self.ref['Absorbance'])
            plt.xlabel(r'Wavenumber (cm^-1)')
            plt.ylabel('Absorbance')
            return fig,ax
            
#poly='//pclasdocnew.univ-lille.fr/Doc/79344/Documents/donnes/donnes femto/DATA RAL MARS 2019 oxford/polystyrene FTIR.CSV '
#ref=pd.read_csv(poly,header=None,names=['Wavelength','Absorbance'])
#ref=ref[(ref['Wavelength']>1250) & (ref['Wavelength']<1700)].reset_index(0,drop=True)
#poly_exp='//pclasdocnew.univ-lille.fr/Doc/79344/Documents/donnes/donnes femto/DATA RAL MARS 2019 oxford/calib_13-03-2019_TRMPS.CSV '
#data=pd.read_csv(poly_exp,header=None,names=['Pixel','Absorbance']) 
#detector1=data[data.iloc[:,0]<len(data)/2]
#detector2=data[data['Pixel']>len(data)/2]
## vowels list
#vowels = ['e', 'a', 'u', 'o', 'i']
#
## sort the vowels
#vowels=list(vowels.sort())
#
## print vowels
#print('Sorted list (in Descending):', vowels)
#
#calibration=Calibration(ref,detector2)
#
#calibration.cutRef((1470,3000))
#calibration.plotDataRef()
#calibration.calibrationFromGrapth()
#calibration.autoCalibrationWithSpectra(min_ref=0.5)
#curve2=calibration.calibration_curve
#
#
#calibration2=Calibration(ref,detector1)
#calibration2.cutRef((1200,1570))
#calibration2.plotDataRef()
#calibration2.calibrationFromGrapth()
##calibration2.autoCalibrationWithSpectra(min_ref=0.5)
#curve1=calibration2.calibration_curve
#
#pixx=np.linspace(0,256-1,256)
#calib1=curve1(pixx[0:128])
#calib2=curve2(pixx[128:])
#calib=np.concatenate((calib1,calib2),axis=None)
#
#
#plt.figure()
#plt.plot(calib, pixx)
#plt.show()