# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:18:25 2020

@author: lucas
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from outils import selectTraces
from MaptplotLibCursor import SnaptoCursor
import pandas as pd
from FormatFigures import FiguresFormating
from PlotSVDClass import PlotSVD

class ExploreData(PlotSVD,FiguresFormating):
    def __init__(self,x,data,wavelength,selected_traces=None,selected_wavelength=None,cmap='viridis',**kwargs):
        units=dict({'time_unit':'ps','time_unit_high':'ns','time_unit_low':'fs','wavelength_unit':'nm','factor_high':1000,'factor_low':1000},**kwargs)
        self.data = data
        self.x = x
        self.wavelength = wavelength
        if selected_traces == None:
            self.selected_traces = data
            self.selected_wavelength = None
        else:
            self.selected_traces = selected_traces
            self.selected_wavelength = selected_wavelength
        self._SVD_fit = False
        self.units = units
        self.color_map = cmap
        PlotSVD.__init__(self,self.x,self.data,self.wavelength,self.selected_traces,self.selected_wavelength)
        
    def plotTraces(self,auto=False,size=14):
        if auto:
            values=[i for i in range(len(self.wavelength))[::round(len(self.wavelength)/11)]]
        else:
            if self.selected_wavelength is not None:
                values=np.where(np.in1d(self.wavelength,self.selected_wavelength))
            else:
                values=[i for i in  range(self.data.shape[1])]
        if len(values) <=10 or auto:
            if self._SVD_fit:
                legenda=['left SV %i' %i for i in range(1,self.data.shape[1]+1)]
            elif self.selected_wavelength is not None:
                legenda=[f'{round(i)} {self.units["wavelength_unit"]}' for i in self.wavelength[values]]
            else:
                legenda=[f'curve {i}' for i in  range(self.data.shape[1])]
        fig,ax = plt.subplots(1,figsize=(11,6))
        alpha=0.60
        for i in values:
            ax.plot(self.x,self.data[:,i],marker='o',alpha=alpha,ms=4,ls='')
        if self.data.shape[1]<=10 or auto:
            ax.legend(legenda,loc='best',ncol=2)
        self._formatFigure(ax,self.data,self.x,size=size)
        self._axisLabels(ax,f'Time ({self.units["time_unit"]})','$\Delta$A',size=size)
        return fig,ax
    
    def plotSpectra(self, times='all',rango=None,n_points=0,cover_range=None,from_max_to_min=True,
                    cmap=None,ncol=1,size=14,legend=True, select=False,select_number=-1,include_rango_max=True): 
        assert times is 'all' or times is 'auto' or type(times) == list, 'times should be either "all" or "auto" \n \
            or a list with selcted point to plot \n \
            or a list this form ["auto", number of spectra(optional; int),  wavelenght to select spectra(optional; int)] \n \
            if times is a list and the first element is "auto" then spectra will be auto plotted \n \
            times should have this form:\n\
            times=["auto", number_of_spectra(optional; int),  wavelenght_to_select_spectra(optional; int)],\n \
            with three possible options:\n\
            1 -->if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths\n \
            2 -->if ["auto",n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all wavelengths\n \
            3 -->if ["auto",n_number_spec,wavelenght] n_number spectra will be plotted equally spaced at the selected wavelenght'
        if cmap is None:
            cmap=self.color_map
        data = self.data
        wavelength=self.wavelength if self.wavelength is not None else np.array([i for i in range(len(data[1]))])
        if data.shape[0]>300 and times=='all':
            times='auto'
            print('More than 300 spectra cannot be plotted or your computer risk of running out of memory')
        times=self._timeToRealTimes(times,rango,include_rango_max,from_max_to_min)
        legenda=['{:.2f}'.format(i*self.units['factor_low']) + ' '+ self.units['time_unit_low'] if abs(i)<0.09
                 else '{:.2f}'.format(i) + ' '+ self.units['time_unit'] if i<999
                 else '{:.2f}'.format(i/self.units['factor_high']) + ' '+ self.units['time_unit_high'] for i in times]
        a=np.linspace(0,1,len(times))
        c=plt.cm.ScalarMappable(norm=[0,1],cmap=cmap)
        colors=c.to_rgba(a,norm=False)
        fig,ax = plt.subplots(1,figsize=(11,6))
        tiempo=pd.Series(self.x)
        for i in range(len(times)):
            index=(tiempo-times[i]).abs().sort_values().index[0]
            if n_points!=0:
                ax.plot(wavelength,np.mean(data[index-n_points:index+n_points,:],axis=0),c=colors[i],label=legenda[i])
            else:
                ax.plot(wavelength,data[index,:],c=colors[i],label=legenda[i])
        self._formatFigure(ax,data,wavelength,size=size,x_tight=True,set_ylim=False)
        self._axisLabels(ax,self._getWaveLabel(),'$\Delta$A',size=size)
        
        if legend:
            leg=plt.legend(loc='best',ncol=ncol,prop={'size': size})
            leg.set_zorder(np.inf)
        else:
            cnorm = Normalize(vmin=times[0],vmax=times[-1])
            cpickmap = plt.cm.ScalarMappable(norm=cnorm,cmap=cmap)
            cpickmap.set_array([])
            plt.colorbar(cpickmap).set_label(label='Time ('+self.time_unit+')',size=15)
        if cover_range is not None:
           self._coverExcitation(ax,cover_range,self.wavelength) 
           
        if select:
            self.cursor = SnaptoCursor(ax, wavelength,wavelength*0.0)
            plt.connect('axes_enter_event', self.cursor.onEnterAxes)
            plt.connect('axes_leave_event', self.cursor.onLeaveAxes)
            plt.connect('motion_notify_event', self.cursor.mouseMove)
        return fig, ax
    
    def plot3D(self,cmap=None):
        if cmap is None:
            cmap=self.color_map
        X=self.x
        Z=self.data.transpose()
        Y=self.wavelength
        if self.wavelength_unit=='cm-1':
            xlabel='Wavenumber (cm$^{-1}$)'
        else:
            xlabel=f'Wavelength ({self.wavelength_unit})'
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(8,4))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                       linewidth=0, antialiased=False)
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.set_xlabel(f'Time ({self.time_unit})')
        ax.set_ylabel(xlabel)
        ax.set_zlabel('$\Delta$A')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig, ax
    
    def selectTraces(self,points=10,average=1, avoid_regions=None):
        if points == 'all':
            self.selected_traces,self.selected_wavelength=self.data,self.wavelength
        else:
            self.selected_traces,self.selected_wavelength = selectTraces(self.data,self.wavelength, points, average, avoid_regions)
    
    def selectTracesGrapth(self,points):
        fig,a=self.plotSpectra(select=True)
        fig.canvas.mpl_connect('close_event', self.selectTraces(self.cursor.datax,points))
    
    def _timeToRealTimes(self,times,rango,include_max,from_max_to_min):
        if times[0]=='auto':
            if len(times)==1:
                times=self._getAutoPoints(rango=rango,include_rango_max = include_max, decrease=from_max_to_min)
            elif len(times)==2:
                times=self._getAutoPoints(spectra=times[1],rango=rango,include_rango_max = include_max, decrease=from_max_to_min)
            elif len(times)==3:
                times=self._getAutoPoints(times[1],times[2],rango=rango,include_rango_max = include_max, decrease=from_max_to_min)   
            else:
                print('if first element is "auto" then spectra will be auto plotted \n \
                      then the list can be only   ["auto"] or:\n\
                      ["auto", number of spectra(optional; int),  wavelenght to select spectra(optional; int)],\n \
                      if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",15,wavelenght] 15 spectra will be plotted equally spaced at the selected wavelenght')
        
        elif times is 'all':
            times=self.x
        elif times is 'auto': 
            times=self._getAutoPoints(rango=rango,include_rango_max=include_max,decrease=from_max_to_min)  
        times=sorted(list(set(times)))
        return times
    
    def _getWaveLabel(self):
        if self.wavelength is None:
            xlabel='pixel'
        elif self.units['wavelength_unit']=='cm-1':
            xlabel='Wavenumber (cm$^{-1}$)'
        else:
            xlabel=f'Wavelength ({self.units["wavelength_unit"]})'
        return xlabel
        
    def _getAutoPoints(self,spectra=8,wave=None,rango=None, include_rango_max=True,decrease=True):
        data=self.data
        wavelength=self.wavelength  
        x=self.x
        if rango is not None:
            assert type(rango) is list, 'rango should be None or a list containing the minimum and maximum value of the range to select points'
            rango=sorted(rango)
            mini=np.argmin([abs(i-rango[0]) for i in x ])
            maxi=np.argmin([abs(i-rango[1]) for i in x ])
            x=x[mini:maxi+1]
            data=data[mini:maxi+1,:]
        if wave is None:
            idx=np.unravel_index(np.argmax(abs(data), axis=None), data.shape)
            if decrease and rango is None:#get points from the maximum of the trace (at the maximum wavelenght) to the end
                point=[idx[0]+np.argmin(abs(data[idx[0]:,idx[1]]-i)) for i in np.linspace(np.min(data[idx[0]:,idx[1]]),np.max(data[idx[0]:,idx[1]]),spectra)]
            else:#get points from the miniimum of the trace (at the maximum wavelenght) to the maximum
                point=[0+np.argmin(abs(data[:,idx[1]]-i)) for i in np.linspace(np.min(data[:,idx[1]]),np.max(data[:,idx[1]]),spectra)]
                if 0 not in point:
                        point[np.argmin(point)]=0
            if rango is not None and include_rango_max:
                     point[np.argmax(point)]=-1
#            print (wavelength[idx[1]])
            return np.sort(np.array(x)[point])
        else:
            if wavelength is not None:
                wave_idx=np.argmin(abs(np.array(wavelength)-wave))
                idx=np.argmax(abs(data[:,wave_idx]))
                if decrease and rango is None:    
                    point=[idx+np.argmin(abs(data[idx:,wave_idx]-i)) for i in np.linspace(np.min(data[idx:,wave_idx]),np.max(data[idx:,wave_idx]),spectra)]
                else:
                    point=[0+np.argmin(abs(data[:,wave_idx]-i)) for i in np.linspace(np.min(data[:,wave_idx]),np.max(data[:,wave_idx]),spectra)]
                    if 0 not in point:
                        point[np.argmin(point)]=0
                if rango is not None and include_rango_max:
                     point[np.argmax(point)]=-1
                print (wavelength[wave_idx])
                return np.sort(np.array(x)[point])
            else:
                print('Wavelenth is not defined')
