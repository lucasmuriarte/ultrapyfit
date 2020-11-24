# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:29:27 2020

@author: lucas
"""
from scipy.sparse.linalg import svds as SVD
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from outils import selectTraces

class PlotSVD:
    def __init__(self,x,data,wavelength,selected_traces=None,selected_wavelength=None):
        self.data=data
        self.x=x
        self.wavelength=wavelength
        self.selected_traces=selected_traces
        self.selected_wavelength=selected_wavelength
        self.S=None
        
    def _calculateSVD(self, vectors=15):
        u,s,v=SVD(self.data,k=vectors)
        return u[:,::-1],s[::-1],v[::-1,:]
        
    def plotSVD(self,vectors=1,select=False,calculate=15):
        wavelength=self.wavelength
        if self.S is None or len(self.S) != calculate :
            self.U,self.S,self.V=self._calculateSVD(vectors=15)
        assert vectors>0 and vectors<len(self.S), 'vector value should be between 1 and the number of calculated values'
        if vectors== 'all':
            vectors=len(self.S)
        self.fig,self.ax=plt.subplots(1,3, figsize=(14,6))
        self.ax[1].plot(range(1,len(self.S)+1),self.S,marker='o')
        for i in range(vectors): 
            self.ax[0].plot(self.x,self.U[:,i])
            self.ax[2].plot(wavelength,self.V[i,:])
        self.ax[0].set_title('Left singular vectors')
        self.ax[1].set_title('Eingen values')
        self.ax[2].set_title('Right singular vectors')
        self.number_of_vectors_plot=vectors
        self.VerticalSVD=self.ax[1].axvline(vectors,alpha=0.5,color='red',zorder=np.inf)  
        axspec = self.fig.add_axes([0.20, .02, 0.60, 0.01],facecolor='orange')
        self.specSVD = Slider(axspec, 'curve number', 1, len(self.S),valstep=1,valinit=vectors)
        self.specSVD.on_changed(self._updatePlotSVD)
        if select != False:
            b_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
            self.button = Button(b_ax, 'Select', color='tab:red', hovercolor='0.975')
            self.button.on_clicked(self._selectSVD)
        self.fig.show()
    
    def plotSingularValues(self,data='all', size=14,log_scale=True):
        if data == 'selected':
            dat=self.selected_traces
        else:
            dat=self.data
        SVD_values=(np.linalg.svd(dat,full_matrices=False,compute_uv=False))**2
        x=np.linspace(1,len(SVD_values),len(SVD_values))
        f, ax=plt.subplots(1)
        plt.plot(x,SVD_values,marker='o',alpha=0.6,ms=4,ls='')
        plt.ylabel('Eigen values', size=size)
        plt.xlabel('number',size=size)
        plt.minorticks_on()
        ax.tick_params(which='both',direction='in',top=True,right=True ,labelsize=size)
        if log_scale:
            plt.yscale("log")
        return f, ax  
    
    def _updatePlotSVD(self,val):
        wavelength=self.wavelength
        value = int(round(self.specSVD.val))
        colores=['tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','tab:blue']
        if value > self.number_of_vectors_plot:
            if value+1==self.number_of_vectors_plot:
                valueC=value*1.0
                if value>10:
                   valueC=value-10*(value//10)  
                self.ax[0].plot(self.x,self.U[:,value],color=colores[valueC+1])
                self.ax[2].plot(wavelength,self.V[value,:],color=colores[valueC+1])
            else:
                for i in range(int(self.number_of_vectors_plot),int(value)):
                    valueC=i
                    if i>10:
                        valueC=i-10*(value//10)  
                    self.ax[0].plot(self.x,self.U[:,i],color=colores[valueC-1])
                    self.ax[2].plot(wavelength,self.V[i,:],color=colores[valueC-1])
            self.VerticalSVD.remove()
            self.VerticalSVD=self.ax[1].axvline(value,alpha=0.5,color='red',zorder=np.inf)      
            self.number_of_vectors_plot=value*1.0
        elif value < self.number_of_vectors_plot:
            self.VerticalSVD.remove()
            self.VerticalSVD=self.ax[1].axvline(value,alpha=0.5,color='red',zorder=np.inf) 
            del self.ax[0].lines[-1]
            del self.ax[2].lines[-1]
            self.number_of_vectors_plot=value*1.0
        else:
            pass
        self.fig.canvas.draw_idle()     

    def _selectSVD(self,val):
        value = int(round(self.specSVD.val))
        self.selected_traces = self.U[:,:value]
        self.selected_wavelength = np.linspace(1,value,value)
        self._SVD_fit=True
        plt.close(self.fig)
    
    def selectTraces(self,space=10,points=1, avoid_regions=None):
        self.selected_traces, self.selected_wavelength = selectTraces(self.data,self.wavelength,space,points, avoid_regions)