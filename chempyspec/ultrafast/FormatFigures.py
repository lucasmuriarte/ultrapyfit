# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:12:16 2020

@author: lucas
"""
import numpy as np
from matplotlib.patches import Rectangle

class FiguresFormating:
    def _coverExcitation(self,ax,x_range,wavelength):
        ymin, ymax = ax.get_ylim()
        ymin = ymin-ymin*0.05
        ymax = ymax - ymax*0.05
        mini=np.argmin([abs(x_range[0]-i) for i in wavelength])
        maxi=np.argmin([abs(x_range[1]-i) for i in wavelength])
        rect = Rectangle([wavelength[mini]-1,ymin],width = wavelength[maxi]-wavelength[mini]+2, 
                                       height=abs(ymax)+abs(ymin),fill=True, color='white',zorder=np.inf)
        ax.add_patch(rect)    
    
    def _axisLabels(self,ax,x_label,y_label,size=14):
        ax.set_ylabel(y_label,size=size)
        ax.set_xlabel(x_label,size=size)
                
    def _formatFigure(self,ax,data,x_vecto,size=14,x_tight=False,set_ylim=True,val=50):
        if set_ylim:
            ax.set_ylim(np.min(data)-abs(np.min(data)*0.1),np.max(data)+np.max(data)*0.1)
        if x_tight:
            ax.set_xlim(x_vecto[0],x_vecto[-1])
        else:    
            ax.set_xlim(x_vecto[0]-x_vecto[-1]/val,x_vecto[-1]+x_vecto[-1]/val)
        ax.axhline(linewidth=1,linestyle='--', color='k')
        ax.ticklabel_format(style='sci',axis='y')
        ax.minorticks_on()
        ax.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)