# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:44:16 2019

@author: 79344
"""
import copy
from scipy.sparse.linalg import svds as SVD
from pylab import pcolormesh
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter as SF
import scipy.integrate as integral
#import matplotlib as mpl
from scipy.special import erf
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons
from lmfit.models import PolynomialModel
from PyQt5.QtGui import  QIcon
from seaborn import distplot
from matplotlib.offsetbox import AnchoredText
from copy import deepcopy

class SnaptoCursor(object):
    def __init__(self, ax,x, y,number_click=-1,vertical_draw=True,draw='snap',color=False,single_line=True):
        if number_click==-1:
            self.number_click=np.inf
        else:
            self.number_click=number_click
        self.ax = ax
        self.draw=draw
        self.vertical_draw=vertical_draw
        self.color=color
        self.x = x
        self.y = y
        self.similar = y==np.zeros(len(y))
        self.datax=[]
        self.datay=[]
        self.scat=[]
        self.single_line=single_line
    def mouseMove(self, event):
        if not event.inaxes: return
        self.x_pos, self.y_pos = event.xdata, event.ydata
        if self.single_line:
            indx = np.searchsorted(self.x, [self.x_pos])[0]
            x = self.x[indx]
            y = self.y[indx]
        else:
            x = self.x_pos
            y = self.y_pos
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        if abs(x)>=0.1:
            texto_x=1
        else:
            try:
                texto_x=[True if i=='0' else False for i in str(x).split('.')[1]].index(False)+1
            except:
                texto_x=3
        if abs(y)>=0.1:
            texto_y=1
        else:
            try:
                texto_y=[True if i=='0' else False for i in str(y).split('.')[1]].index(False)+1
            except:
                texto_y=3
        if self.similar.all()==False:
            self.lx.set_ydata(y)
            self.txt.set_text('x='+str(round(x,texto_x))+', y='+str(round(y,texto_y)))
            self.txt.set_position((x,y))
        else:
            self.txt.set_text('x=' +str(round(x,texto_x)))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()
    
    def onClick(self,event):
        if not event.inaxes: return
        if event._button_svd_select==1:
            #print(self.number_click)
            if len(self.datax)<self.number_click:
                x,y=event.xdata,event.ydata
                if self.draw=='snap':   
                    indx = np.searchsorted(self.x, [x])[0]
                    x = self.x[indx]
                    y = self.y[indx]
                self.datax.append(x)
                self.datay.append(y)
#                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#                      ('double' if event.dblclick else 'single', event.button,
#                       event.x, event.y, event.xdata, event.ydata))
                if self.vertical_draw:
                    self.scat.append(self.ax.axvline(self.datax[-1],alpha=0.5,color='red',zorder=np.inf))
                else:
                    self.scat.append(self.ax.scatter(self.datax,self.datay, color='red',marker='x',zorder=np.inf))
            else:
                pass
            self.ax.figure.canvas.draw_idle()
        elif event._button_svd_select==3:
            if len(self.datax)==0:
                pass
            else:
                del self.datax[-1]
                del self.datay[-1]
                self.scat[-1].remove()
                del self.scat[-1]
                self.ax.figure.canvas.draw_idle()  
    
    def onEnterAxes(self,event):
        if not event.inaxes: return
        try:
            self.onLeaveAxes(event)
        except:
            pass
        if self.similar.all()==False:
            self.lx = self.ax.axhline(color='k',alpha=0.2)  # the horiz line
        if self.single_line:
            try:
                line=self.ax.lines[0]
                self.x=line.get_xdata()
                if self.similar.all()==False:
                    self.y=line.get_ydata()
            except:
                pass
        self.ly = self.ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = self.ax.plot([0],[0], marker="o", color="crimson", zorder=3)
        self.txt = self.ax.text(0.7, 0.9, '')
        if self.color is not False:
            event.inaxes.patch.set_facecolor(self.color)
        event.canvas.draw()
    
    def onLeaveAxes(self,event):
        if not event.inaxes: return
        #print ('leave_axes', event.inaxes)
        self.marker.remove()
        self.ly.remove()
        self.txt.remove()
        if self.similar.all()==False:
            self.lx.remove()
        event.inaxes.patch.set_facecolor('white')
        event.canvas.draw()
        
#class SnaptoCursor(object):
#    def __init__(self, ax, x, y):
#        self.ax = ax
#        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
#        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
#        self.x = x
#        self.y = y
#        self.txt = ax.text(0.7, 0.9, '')
#        
#    def mouse_move(self, event):
#        if not event.inaxes: return
#        x, y = event.xdata, event.ydata
#        indx = np.searchsorted(self.x, [x])[0]
#        x = self.x[indx]
#        y = self.y[indx]
#        self.ly.set_xdata(x)
#        self.marker.set_data([x],[y])
#        self.txt.set_text('x=%1.2f' % (x))
#        self.txt.set_position((x,y))
#        self.ax.figure.canvas.draw_idle()

        
class GlobalFit(lmfit.Minimizer):
    def __init__(self,x,data, wavelength=None,exp_no=1,excitation=None,deconv=True,path=None,**kwargs):
        derivate_space=dict({'derivate':False,'window_length':25,'polyorder':3,'deriv':1,'done':False},**kwargs)
        self.global_fit_version=2.2
        self.working_directory=path
        self.deconv=deconv
        self.tau_inf=1E+12 #time infinite used for femto long live species
        self.x=np.array(x)
        self.original_data=data
        self.original_x=np.array(x)
        self.original_wavelength=np.array(wavelength)
        self.exp_no=exp_no
        if derivate_space['derivate']:
            self.report_deriv=', '.join([f'{key}: {derivate_space[key]}' for key in derivate_space ])
            data2=0.0*data
            if derivate_space['done']:
                self.derivative_space=derivate_space
                self.data=data
            else:
                for i in range(len(data)):
                    data2[i,:]=SF(data[i,:],
                              window_length=derivate_space['window_length'], 
                              polyorder=derivate_space['polyorder'],deriv=derivate_space['deriv'])
                    self.data=data2
                    self.derivative_space=derivate_space
        else:
            self.report_deriv='\tFalse'
            self.data=data
            self.derivative_space=False
        self.excitation=excitation
        self.all_fit={}
        self.fit_number=0
        self.curve_resultados=0.0*self.data[:]
        if wavelength is None:
            self.wavelength=wavelength
            general_cal='\tNone'
        else:
            self.wavelength=np.array(wavelength)
            general_cal='\tAlready calibrated'
        self.save={'path':self.working_directory,'name':'','format':'png','dpi':300}
        self.time_unit='ps'
        self.time_unit_high='ns'
        self.time_unit_low='fs'
        self.factor_high=1000
        self.factor_low=1000
        self.wavelength_unit='nm'
        self.prefit_done=False
        self.params_initialized=False
        self.Fit_completed=False
        self.S=None#for singular value decomposition
        self.V=None
        self.U=None
        self.SVD_fit=False
        self.BK7_param={'b1':1.03961212,'b2':0.231792344,'b3':1.01046945,'c1':6.00069867e10-3,'c2':2.00179144e10-2,'c3':103.560653}
        self.SiO2_param={'b1':0.69616,'b2':0.4079426,'b3':0.8974794,'c1':4.67914826e10-3,'c2':1.35120631e10-2,'c3':97.9340025}
        self.CaF2_param={'b1':0.5675888,'b2':0.4710914,'b3':38.484723,'c1':0.050263605,'c2':0.1003909,'c3':34.649040}  
        self.alpha={'activate':False,'value':0.3,'size':0.5}
        self.general_report={'File':'\tNone','Excitation':excitation,'Units':{'Time unit':'\tps','Wavelength unit':'nm'},
                             'Data Shape':{'Initial number of traces':f'{data.shape[1]}','Initial time points':f'{data.shape[0]}','Actual number of traces':'All','Actual time points':'All'},
                             'Preprocessing':{'Calibration':general_cal,'GVD correction':[],'IRF Fit':'\tNone','Baseline correction':None,
                             'Cutted Wavelengths':[],'Cutted Times':[],'Deleted wavelength points':[],
                             'Deleted time points':[],'Average time points':[],'Time shift':[],'Polynom fit':[],'Derivate data':self.report_deriv},'Fits done':{},'Sequence of actions':[],'User comments':[]}
        
        self.type_fit='Exponential'
        self.original_preprocessing=self.general_report['Preprocessing']
        self.inner_cut_done=None
        self.excitation_width=11
        self.single_fits={}
        self.bootstrap_residues_record={}
        self.conf_interval={}
        self.target_models={}
        self.dispersion_Caf2=0
        self.dispersion_SiO2=0
        self.dispersion_BK7=0
        self.data_before_first_selection=None
        self.data_before_cut=None
        self.data_before_bg=None
        self.data_before_GVD=None
        self.data_before_deriv=None
        self.data_before_time_cut=None
        self.data_before_average_time=None
        self.data_before_flutuation_correction=None
        self.data_before_del_point=None
        self.x_before_time_cut=None
        self.NO_deconv_all_traces_same_cero=True
        self.GVD_correction=False
        self.experiment_manual_load=''
        self.gvd_Grapth=False
        self.IRF_value=None
        self.number_it=0
        self.qt_path=None
        self.average_trace=0
        self.weights={'apply':False,'vector':None,'range':[],'type':'constant','value':2}
        self.color_map='viridis'
        super().__init__(self.objective, self.parametros(), nan_policy='propagate')
    
    def printGeneralReport(self):
        texto=[]
        for main_key in self.general_report.keys():
            texto.append(f'--------------------\n')
            texto.append(f'{main_key}: ')
            if type(self.general_report[main_key]) is str and type(self.general_report[main_key]) is not None :
                texto.append(f'\t{self.general_report[main_key]}\n')
            elif type(self.general_report[main_key]) is int or type(self.general_report[main_key]) is float:
                texto.append(f'{str(self.general_report[main_key])}\n')
            elif type(self.general_report[main_key]) is list:
                if len(self.general_report[main_key])==0:
                    if len(main_key)<=15:
                        texto.append(f'\t\tNone\n')
                    else:
                        texto.append(f'\tNone\n')
                else:
                    if type(self.general_report[main_key][0]) is int or type(self.general_report[main_key][0]) is float:
                        a=[str(i) for i in self.general_report[main_key]]
                        texto.append('\t'+', '.join(a)+'\n')
                    elif type(self.general_report[main_key][0]) is str:
                        texto.append('\t\n')
                        for i in self.general_report[main_key]:
                           texto.append(i)
                           texto.append('\t\n')
                    else:
                        try:
                            texto.append(f'\t{self.general_report[main_key]}\n')
                        except:
                            pass
            elif type(self.general_report[main_key]) is dict:
                texto.append('\n')
                dictionary=self.general_report[main_key]
                for sub_key in dictionary.keys():
                    texto.append(f'\t{sub_key}: ')
                    if type(dictionary[sub_key]) is str and type(dictionary[sub_key]) is not None :
                       texto.append(f'\t{self.general_report[main_key][sub_key]}\n')
                    elif type(dictionary[sub_key]) is int or type(dictionary[sub_key]) is float:
                        texto.append(f'\t\t{str(self.general_report[main_key][sub_key])}\n')
                    elif type(dictionary[sub_key]) is list:
                        if len(dictionary[sub_key])==0:
                            if len(sub_key)<=15:
                                texto.append(f'\t\tNone\n')
                            else:
                                texto.append(f'\tNone\n')
                        else:
                            if type(dictionary[sub_key][0]) is int or type(dictionary[sub_key][0]) is float:
                                a=[str(i) for i in dictionary[sub_key]]
                                texto.append('\t'+', '.join(a)+'\n')
                            elif type(dictionary[sub_key][0]) is str:
                                texto.append('\t\n')
                                for i in dictionary[sub_key]:
                                   texto.append(i)
                                   texto.append('\t\n')
                            else:
                                try:
                                    texto.append(f'\t{dictionary[sub_key]}\n')
                                except:
                                    pass
                    else:
                        try:
                            texto.append(f'\t{self.general_report[main_key][sub_key]}\n')
                        except:
                            pass
            else:
                try:
                    texto.append(f'\t{self.general_report[main_key]}\n')
                except:
                    pass
            texto.append('\n')
        return ''.join(texto)
                
    def reinitiate(self):
        self.type_fit='Exponential'
        self.inner_cut_done=None
        self.deconv=True
        self.tau_inf=1E+12 #time infinite used for femto long live species
        self.excitation=None
        self.all_fit={}
        self.bootstrap_residues_record={}
        self.conf_interval={}
        self.target_models={}
        self.single_fits={}
        self.fit_number=0
        self.curve_resultados=0.0*self.data[:]
        self.time_unit='ps'
        self.time_unit_high='ns'
        self.time_unit_low='fs'
        self.wavelength_unit='nm'
        self.factor_high=1000
        self.factor_low=1000
        self.prefit_done=False
        self.params_initialized=False
        self.Fit_completed=False
        self.dispersion_Caf2=0
        self.dispersion_SiO2=0
        self.dispersion_BK7=0
        self.data_before_first_selection=None
        self.data_before_cut=None
        self.data_before_bg=None
        self.data_before_GVD=None
        self.data_before_deriv=None
        self.data_before_time_cut=None
        self.data_before_average_time=None
        self.data_before_flutuation_correction=None
        self.x_before_time_cut=None
        self.S=None#for singular value decomposition
        self.V=None
        self.U=None
        self.SVD_fit=False
        self.weights={'apply':False,'vector':None,'range':[],'type':'constant','value':2}
        if self.wavelength is not None:
            general_cal='None'
        else:
            general_cal='Already calibrated'
        self.derivative_space=False
        self.NO_deconv_all_traces_same_cero=True
        self.GVD_correction=False
        self.gvd_Grapth=False
        self.number_it=0
        self.general_report=self.original_general_report={'File':'\tNone','Excitation':self.excitation,'Units':{'Time unit':'\tps','Wavelength unit':'nm'},
                             'Data Shape':{'Initial number of traces':f'{self.original_data.shape[1]}','Initial time points':f'{self.original_data.shape[0]}','Actual number of traces':'All','Actual time points':'All'},
                             'Preprocessing':{'Calibration':general_cal,'GVD correction':[],'IRF Fit':'\tNone','Baseline correction':'None',
                             'Cutted Wavelengths':[],'Cutted Times':[],'Deleted wavelength points':[],
                             'Deleted time points':[],'Average time points':[],'Time shift':0,'Polynom fit':[],'Derivate data':self.report_deriv},'Fits done':{},'Sequence of actions':[],'User comments':[]}
    
    def defineWeights(self,rango,typo,val=2):
        '''typo should be a string exponential or r_expoential or parabol or a int
        exmaple:
        constant value 5, [1,1,1,1,...5,5,5,5,5,....1,1,1,1,1]
        exponential for val= 2 [1,1,1,1,....2,4,9,16,25,....,1,1,1,] 
                    for val= 3 [1,1,1,1,....3,8,27,64,125,....,1,1,1,]
        r_expoential [1,1,1,1,...25,16,9,4,2,...1,1,1,]
        exp_mix [1,1,1,1,...2,4,9,4,2,...1,1,1,]'''
        x=self.x
        rango=sorted(rango)
        if typo == 'constant':
            weight=[val if i >rango[0] and i < rango[1] else 1 for i in x]
        else:
            mini=np.argmin([abs(i-rango[0]) for i in x ])
            maxi=np.argmin([abs(i-rango[1]) for i in x ])
            if typo == 'exponential':
                weight=[1 for i in x[:mini]] +[i**val for i in range(1,maxi-mini+2)] +[1 for i in x[maxi+1:]]
                weight[mini]=val
            elif typo == 'r_exponential':
                weight=[1 for i in x[:mini]] +[i**val for i in range(maxi-mini+1,1,-1)]+[1 for i in x[maxi:]]
                weight[maxi]=val
            elif typo == 'exp_mix':
                if (maxi-mini) % 2 == 0:
                    weight=[1 for i in x[:mini]] +[i**val for i in range(1,(maxi-mini+2)//2)]+[i**2 for i in range((maxi-mini+2)//2,1,-1)]+[1 for i in x[maxi:]]
                else:
                    weight=[1 for i in x[:mini]] +[i**val for i in range(1,(maxi-mini+3)//2)]+[i**2 for i in range((maxi-mini+2)//2,1,-1)]+[1 for i in x[maxi:]]
                weight[mini]=val 
                weight[maxi]=val
        self.weights['vector']=np.array(weight)
        self.weights['type']=typo
        self.weights['range']=rango
        self.weights['value']=val
    
    def flutuationsCorrectionGraph(self,order=3):
        if self.data_before_first_selection is not None:
            wavelength=self.wavelength_before_first_selection
        else:
            wavelength=self.wavelength 
        def fitPol(event):
            points=sorted(fluCursor.datax[:-1])
            self.correctionFlutuationsPoly(points,order)
        fig,ax=self.plotSpectra(times='auto')
        plt.title('The number of point should be greater than the polynom order')
        fluCursor=SnaptoCursor(ax,wavelength,wavelength*0.0)
        fig.canvas.mpl_connect('button_press_event', fluCursor.onClick)
        fig.canvas.mpl_connect('motion_notify_event', fluCursor.mouseMove)
        fig.canvas.mpl_connect('axes_enter_event', fluCursor.onEnterAxes)
        fig.canvas.mpl_connect('axes_leave_event', fluCursor.onLeaveAxes)
        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Correct', color='tab:red', hovercolor='0.975')
        button.on_clicked(fitPol)
        fig.show()
             
    def correctionFlutuationsPoly(self,points,order=3):
        assert len(points)>order, 'The number of points need to be higher than the polynom order'
        if self.data_before_first_selection is not None:
            data=self.data_before_first_selection
            wavelength=self.wavelength_before_first_selection
        else:
            data=self.data
            wavelength=self.wavelength  
        if self.data_before_flutuation_correction is None:
            self.data_before_flutuation_correction=data*1.0
            self.preprocess_before_flutuation_correction=self.general_report['Preprocessing']
        n_r,n_c=data.shape
        index=[np.argmin(abs(wavelength-i)) for i in points]
        data_corr=data*1.0
        for i in range(n_r):
            print(i)
            polynom=np.poly1d(np.polyfit(wavelength[index],data[i,index], order))
            data_corr[i,:]=data[i,:]-polynom(wavelength)
        points_str=[str(i) for i in points]
        if self.data_before_first_selection is not None:
            self.data_before_first_selection=data_corr
        else:
            self.data=data_corr
        self.general_report['Preprocessing']['Polynom fit']=f'order {order} polynom fit points: {(",").join(points_str)}'
        self.general_report['Sequence of actions'].append('\t--> Polynom fit for baseline fluctuations')
    
    def shitTime(self,value):
        self.x=self.x-value
        self.general_report['Preprocessing']['Time shift'] += -value
    
    def baselineSubstraction(self,nuber_spec=2,only_one=False):
        data_changed=False
        if nuber_spec==0:
            only_one=True
        if self.data_before_first_selection is not None:
            previous_data=self.data
            previous_wave=self.wavelength
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
            data_changed=True
        if self.data_before_bg is None:
            self.preprocess_before_bg=self.general_report['Preprocessing']
            self.data_before_bg=self.data
            self.x=self.x
            self.wavelength_before_bg=self.wavelength
        if only_one:
            number=np.array(self.data[nuber_spec,:])
            for i in range(len(self.data)):
                self.data[i,:]=self.data[i,:]-number
            self.general_report['Preprocessing']['Baseline correction']=f'Substracted spectrum {nuber_spec}'
            self.general_report['Sequence of actions'].append('\t--> Baseline Correction')
        else:
            mean=np.mean(self.data[:nuber_spec,:],axis=0)
            for i in range(len(self.data)):
                self.data[i,:]=self.data[i,:]-mean
            self.general_report['Preprocessing']['Baseline correction']=f'Substracted the average of the first {nuber_spec} spectra'
            self.general_report['Sequence of actions'].append('\t--> Baseline Correction')
        if data_changed:
            self.data=previous_data
            self.wavelength=previous_wave
            
    def createNewDir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.working_directory=path
        self.save['path']=self.working_directory
        
    def defineUnits(self,time,wavelength):
        times=['Ato s','fs','ps','ns','μs','ms','s','min','h']
        assert (time in times[1:-1]) or time=='us'
        assert type(wavelength) == str
        if time=='us':
            self.time_unit='µs'
            self.time_unit_high='ms'
            self.time_unit_low='ns'
        else:
            index=times.index(time)
            self.time_unit=time
            self.time_unit_high=times[index+1]
            self.time_unit_low=times[index-1]
        if self.time_unit=='s':
            self.factor_high=60
        elif self.time_unit=='min':
            self.factor_high=60
            self.factor_low=60
        else:
            pass
        self.wavelength_unit=wavelength
        self.general_report['Units']['Time unit']=f'\t{self.time_unit}'
        self.general_report['Units']['Wavelength unit']=self.wavelength_unit
        self.general_report['Sequence of actions'].append('\t--> Units changed')
        
    def derivateSpace(self,window_length=25,polyorder=3,deriv=1,mode='mirror'):
        data_changed=False
        if self.data_before_first_selection is not None:
            previous_data=self.data
            previous_wave=self.wavelength
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
            data_changed=True
        if self.data_before_deriv is None:
            self.preprocess_before_deriv=self.general_report['Preprocessing']
            self.data_before_deriv=self.data
            self.x=self.x
            self.wavelength_before_deriv=self.wavelength
        data2=0.0*self.data
        for i in range(len(self.data)):
            data2[i,:]=SF(self.data[i,:],window_length=window_length, polyorder=polyorder,deriv=deriv,mode=mode)
        self.derivative_space={'derivate':True,'window_length':window_length,'polyorder':polyorder,'deriv':deriv}
        self.data=data2
        self.general_report['Preprocessing']['Derivate data']='\t'+'\t\t\t'.join([f'{key}: {self.derivative_space[key]}\n' for key in self.derivative_space])
        self.general_report['Sequence of actions'].append('\t--> Derivation of Data')
        if data_changed:
            self.data_before_first_selection=data2
            self.data=previous_data
            self.wavelength=previous_wave
            
    def selectTraces(self,space=10,points=1,avoid_excitation=9, avoid_regions=None):
        """ cut the data in time range
         Parameters
        ----------
        space: type=int or list or "auto": if type(int) a series of traces separated by the value indicated will be selected.
                                         if type(list) the traces in the list will be selected. 
                                         if auto, 10 sperated spectra will be selected
        points:bining point sourranding the selected wavelengths
        avoid_excitation: Int self.excitation and self.wavelength should be indicated
                            in the instanciated class the indicated value will be avoid
                            from left or right of the excitation.
        """
        if self.data_before_first_selection is None:
            self.data_before_first_selection=self.data
            self.wavelength_before_first_selection=self.wavelength
        else:
            self.data=self.data_before_first_selection
            self.curve_resultados=self.data*0.0
            self.wavelength=self.wavelength_before_first_selection
        dat=pd.DataFrame(self.data)
        wavelengths=pd.Series([float(i) for i in dat.columns])
        if space is 'auto':
           values=[i for i in range(len(self.wavelength))[::round(len(self.wavelength)/11)]] 
           values=values[1:]
        if type(space) is int:
            if self.wavelength is not None:
                wavelength_unit=1/((self.wavelength[-1]-self.wavelength[0])/len(self.wavelength))
                space=round(space*wavelength_unit)
            first=wavelengths.iloc[0+points]
            values=[first+space*i for i in range(len(wavelengths)) if first+space*i < wavelengths.iloc[-1]]
        elif type(space) is list:
            values=[np.argmin(abs(self.wavelength-i)) for i in space]
        selected_traces=[(wavelengths-values[i]).abs().sort_values().index[0] for i in range(len(values))]
        if self.inner_cut_done is not None:
            excitation_wavelength=np.where((self.wavelength > self.inner_cut_done[0]) & (self.wavelength < self.inner_cut_done[1]))[0]
            selected_traces=[i for i in selected_traces if i not in excitation_wavelength] 
        if type(avoid_excitation) is int and self.inner_cut_done is None:
            assert self.excitation is not None, 'Please indicate excitation wavelenght'
            assert self.wavelength is not None, 'Please indicate wavelenght of experiment'
            excitation_wavelength=np.where((self.wavelength > self.excitation-avoid_excitation) & (self.wavelength < self.excitation+avoid_excitation))[0]
            selected_traces=[i for i in selected_traces if i not in excitation_wavelength]            
        if avoid_regions is not None: 
            assert type(avoid_regions) is list, 'Please regions should be indicated as a list'
            if type(avoid_regions[0]) is not list:
                avoid_regions=[avoid_regions]
            for i in avoid_regions:
                assert len(i) is 2,  'Please indicate 2 number to declare a region'
                i=sorted(i)
                excitation_wavelength=np.where((self.wavelength > i[0]) & (self.wavelength < i[1]))[0]
                selected_traces=[i for i in selected_traces if i not in excitation_wavelength]  
        if points == 0:
            dat=pd.DataFrame(data=[dat.iloc[:,i] for i in selected_traces],
                                  columns=dat.index,index=[str(i+wavelengths[0]) for i in selected_traces]).transpose()
        else:
            if self.inner_cut_done is not None:
                min_indexes=[excitation_wavelength[-1] if i-points > excitation_wavelength[0] and i-points < excitation_wavelength[-1] else i-points for i in selected_traces]
                max_indexes=[excitation_wavelength[0] if i+points > excitation_wavelength[0] and i+points < excitation_wavelength[-1] else i+points for i in selected_traces]
                dat=pd.DataFrame(data=[dat.iloc[:,min_index:max_index].mean(axis=1) for min_index,max_index in zip(min_indexes,max_indexes)],
                                  columns=dat.index,index=[str(i+wavelengths[0]) for i in selected_traces]).transpose()
            else:
                dat=pd.DataFrame(data=[dat.iloc[:,i-points:i+points].mean(axis=1) for i in selected_traces],
                                  columns=dat.index,index=[str(i+wavelengths[0]) for i in selected_traces]).transpose()
        if points == 0:
            self.average_trace=0
        elif points%2 == 0:
            self.average_trace=points*2+1
        else:
            self.average_trace=points*2
        self.data=dat.values
        self.curve_resultados=0.0*self.data[:]
        self.selected_traces=selected_traces
        self.SVD_fit=False
        if self.wavelength is not None:
            wavelengths=pd.Series(self.wavelength)
            self.wavelength=np.array([wavelengths.iloc[i] for i in selected_traces])
        if  self.params_initialized==True:
            self.paramsAlready()
            
    def calculateSVD(self, vectors=15):
        if self.data_before_first_selection is None:
            data=self.data*1.0
        else:
            data=self.data_before_first_selection*1.0
        u,s,v=SVD(data,k=vectors)
        return u[:,::-1],s[::-1],v[::-1,:]
        
    def plotSVD(self,vectors=1,select=False):
        if self.data_before_first_selection is None:
            wavelength=self.wavelength
        else:
            wavelength=self.wavelength_before_first_selection
        if self.S is None:
            self.U,self.S,self.V=self.calculateSVD(vectors=15)
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
        self.specSVD.on_changed(self.updatePlotSVD)
        if select:
            b_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
            self.button = Button(b_ax, 'Select', color='tab:red', hovercolor='0.975')
            self.button.on_clicked(self.selectSVD)
        self.fig.show()
        
    def updatePlotSVD(self,val):
        if self.data_before_first_selection is None:
            wavelength=self.wavelength
        else:
            wavelength=self.wavelength_before_first_selection
        value = int(round(self.specSVD.val))
        colores=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
        if value > self.number_of_vectors_plot:
            valueC=value
            if value>10:
               valueC=value-10*(value//10)  
            self.VerticalSVD.remove()
            self.VerticalSVD=self.ax[1].axvline(value,alpha=0.5,color='red',zorder=np.inf)  
            self.ax[0].plot(self.x,self.U[:,value],color=colores[valueC-1])
            self.ax[2].plot(wavelength,self.V[value,:],color=colores[valueC-1])
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
    
    def selectSVD(self,val):
        if self.data_before_first_selection is None:
            self.data_before_first_selection=self.data
            self.wavelength_before_first_selection=self.wavelength
        value = int(round(self.specSVD.val))
        self.data=self.U[:,:value]
        self.curve_resultados=self.data*0.0
        self.SVD_fit=True
#        self.SVD_wavelength=['left SV %i' %i for i in range(1,value+1)]
        plt.close(self.fig)
            
    
    def plotSingularValues(self, size=14,log_scale=True,save=False):
        SVD_values=(np.linalg.svd(self.data,full_matrices=False,compute_uv=False))**2
        x=np.linspace(1,len(SVD_values),len(SVD_values))
        if log_scale:
            f, ax=plt.subplots(1)
            plt.plot(x,SVD_values,marker='o',alpha=0.6,ms=4,ls='')
            plt.yscale("log")
            plt.ylabel('Eigen values', size=size)
            plt.xlabel('number',size=size)
            plt.minorticks_on()
            ax.tick_params(which='both',direction='in',top=True,right=True ,labelsize=size)
        else:
            f, ax=plt.subplots(1)
            plt.plot(x,SVD_values,marker='o',alpha=0.6,ms=4,ls='')
            plt.ylabel('Eigen values', size=size)
            plt.xlabel('number',size=size)
            plt.minorticks_on()
            ax.tick_params(which='both',direction='in',top=True,right=True ,labelsize=size)
        
        if save:
            if self.save['name']=='':
                    self.save['name']='SVD'
            plt.savefig(self.save['path']+self.save['name']+'.'+self.save['format'], dpi=self.save['dpi'])
            self.save['name']=''
        return f, ax  
        
    def expGaussConv(self,time,tau,sigma):
        return 0.5*np.exp(-tau*time + sigma**2*tau**2/2 )*(1+erf((time-sigma**2*tau)/(sigma*2**0.5)))
    
    def expNGauss (self,time,y0,t0,fwhm,yinf,values,fit_number=None):
        """values should be a list of list containing the pre_exps and taus values"""
        if fit_number is not None:
            tau_inf=self.all_fit[fit_number][6]
        else:
            tau_inf=self.tau_inf
        if tau_inf is not None:
            return y0+sum([(pre_exp)*self.expGaussConv(time-t0,1/tau,fwhm/2.35482) for pre_exp,tau in values])\
                    +(yinf)*self.expGaussConv(time-t0,1/tau_inf,fwhm/2.35482)
        else:
            return y0+sum([(pre_exp)*self.expGaussConv(time-t0,1/tau,fwhm/2.35482) for pre_exp,tau in values])
    
    def expNGaussDataset(self,params, i,fit_number=None):
        """calc 2 exponetial function from params for data set i
        using simple, hardwired naming convention"""
        if fit_number is not None:
            x=self.all_fit[fit_number][0]
            exp_no=self.all_fit[fit_number][4]
            tau_inf=self.all_fit[fit_number][6]
        else:
            tau_inf=self.tau_inf
            exp_no=self.exp_no
            x=self.x
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        fwhm= params ['fwhm_%i' % (i+1)].value
        if tau_inf is not None:
            yinf = params['yinf_%i' % (i+1)].value
        else:
            yinf=None
        values=[[params['pre_exp%i_' % (ii+1)+str(i+1)].value,params['tau%i_' %(ii+1)+str(i+1)].value] for ii in range(exp_no)]            
        return self.expNGauss(x,y0,t0,fwhm,yinf,values,fit_number=fit_number)
           
    def exp1(self,x, tau):
        "basic gaussian"
        return np.exp(-x/tau) 

    def expN (self,time,y0,t0,values):
        """values should be a list of list containing the pre_exps and taus values"""
        return y0+sum([pre_exp*self.exp1(time-t0,tau) for pre_exp,tau in values])
                 
    def expNDataset(self,params,i,fit_number=None):
        """calc 2 exponetial function from params for data set i
        using simple, hardwired naming convention"""
        if fit_number is not None:
            x=self.all_fit[fit_number][0]
            exp_no=self.all_fit[fit_number][4]
        else:
            exp_no=self.exp_no
            x=self.x
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        index=np.argmin([abs(i-t0) for i in x])
        values=[[params['pre_exp%i_' % (ii+1)+str(i+1)].value,params['tau%i_' %(ii+1)+str(i+1)].value] for ii in range(exp_no)]            
        return self.expN(x[index:],y0,t0,values)
    
    def expNDatasetFast(self,params,i,expvects):
        y0 = params['y0_%i' % (i+1)].value
        pre_exp=[params['pre_exp%i_' % (ii+1)+str(i+1)].value for ii in range(self.exp_no)] 
        return y0+sum([pre_exp[iii]*expvects[iii] for iii in range(self.exp_no)])   
    
    def objective(self,params):
        """ calculate total residual for fits to several data sets held
        in a 2-D array"""
        ndata, nx = self.data_before_last_Fit.shape
        resid = 0.0*self.data_before_last_Fit[:]
        # make residual per data set
        if self.type_fit == 'Exponential':
            if self.deconv:
                if(self.GVD_correction): #assumes that t0 are equal to all  wavelengths, and also fwhm's are also the the same for all wavelengths!
                    self.optimized='Deconvolution and t0 fix because GVD was corrected'
                    t0 = params['t0_1'].value
                    fwhm= params ['fwhm_1'].value
                    values=[params['tau%i_1' %(ii+1)].value for ii in range(self.exp_no)] 
                    if self.tau_inf is not None:
                        expvects = [self.expGaussConv(self.x-t0,1/tau,fwhm/2.35482) for tau in values]+[self.expGaussConv(self.x-t0,1/self.tau_inf,fwhm/2.35482)]
                    else:
                        expvects = [self.expGaussConv(self.x-t0,1/tau,fwhm/2.35482) for tau in values]
                    for i in range(nx):
                        resid[:, i] = self.data_before_last_Fit[:, i] - self.expNGaussDatasetFast(params, i, expvects)                
                        if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector']
                else:
                    self.optimized='Deconvolution and t0 vary Due to uncorrected GVD'
                    for i in range(nx):
                        resid[:, i] = self.data_before_last_Fit[:, i] - self.expNGaussDataset(params, i)
                        if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector']
            else:
                if self.NO_deconv_all_traces_same_cero==False:
                    self.optimized='No de convolution and t0 different for each trace'
                    for i in range(nx):
                        resid[:, i] = self.data_before_last_Fit[:, i] - self.expNDataset(params, i)
                        if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector']   
                else:
                    self.optimized='No de convolution and t0 same for all traces'
                    t0 = params['t0_1'].value
                    index=np.argmin([abs(i-t0) for i in self.x])
                    resid = 0.0*self.data_before_last_Fit[index:,:]
                    values=[params['tau%i_1' %(ii+1)].value for ii in range(self.exp_no)] 
                    expvects=[self.exp1(self.x[index:]-t0,tau) for tau in values]
                    for i in range(nx):
                        resid[:, i] = self.data_before_last_Fit[index:, i] - self.expNDatasetFast(params, i, expvects)
                        if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector'][index:]   
        else:
    #        take kmatrix shit from params and solve eqs
            ksize = self.exp_no #size of the matrix = no of exponenses = no of species
            kmatrix = np.array([[params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
            cinitials = [params['c_%i' % (i+1)].value for i in range(ksize)]
            eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
            #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
            eigenmatrix = np.array(vects) 
            coeffs = np.linalg.solve(eigenmatrix, cinitials) #solve the initial conditions sheet
            
            if self.deconv:
                if self.GVD_correction:
                    t0 = params['t0_1'].value
                    fwhm = params['fwhm_1'].value
                    expvects = [coeffs[i]*self.expGaussConv(self.x-t0,-eigs[i],fwhm/2.35482) for i in range(len(eigs))] 
                    concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
                    for i in range(nx):
                        resid[:, i] = self.data_before_last_Fit[:, i] - self.expNGaussDatasetFast(params, i, concentrations) 
                        if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector']
             
                else:  #didnt tested but should work, if no then probably minor correction is needed.
                    for i in range(nx):                
                        t0 = params['t0_%i' % (i+1)].value
                        fwhm = params['fwhm_1'].value
                        expvects = [coeffs[i]*self.expGaussConv(self.x-t0,-eigs[i],fwhm/2.35482) for i in range(len(eigs))] 
                        concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]    
                        resid[:, i] = self.data_before_last_Fit[:, i] - self.expNGaussDatasetFast(params, i, concentrations) 
                        if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector']
            else:                             
                t0 = params['t0_1'].value
                index=np.argmin([abs(i-t0) for i in self.x])
                resid = 0.0*self.data_before_last_Fit[index:,:]
    #            fwhm = 0.0000001 #lets just take very short IRF, effect should be the same
                expvects = [coeffs[i]*self.exp1(self.x[index:]-t0,-eigs[i]) for i in range(len(eigs))] 
                concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
                for i in range(nx):  
                    resid[:, i] = self.data_before_last_Fit[index:, i] - self.expNDatasetFast(params, i, concentrations)          
                    if self.weights['apply']==True: 
                            resid[:, i]=resid[:, i]*self.weights['vector'][index:]
        self.number_it=self.number_it+1
        if(self.number_it % 100 == 0):
            print(self.number_it)
            print(sum(np.abs(resid.flatten())))
        return resid.flatten()
       
    def expNGaussDatasetFast(self, params, i, expvects):
        y0 = params['y0_%i' % (i+1)].value
        pre_exp=[params['pre_exp%i_' % (ii+1)+str(i+1)].value for ii in range(self.exp_no)] 
        if self.tau_inf is not None:
            yinf = params['yinf_%i' % (i+1)].value
            return y0+sum([pre_exp[iii]*expvects[iii] for iii in range(self.exp_no)])+yinf*expvects[-1]
        else:
            return y0+sum([pre_exp[iii]*expvects[iii] for iii in range(self.exp_no)])        
        
    def results(self,result_params,fit_number=None,verify_SVD_fit=False):
        if fit_number is not None:
            x=self.all_fit[fit_number][0]
            #verify type of fit is: either fit to Singular vectors or global fit to traces
            if self.all_fit[fit_number][9] and verify_SVD_fit ==False: #verify type of fit
                data=self.all_fit[fit_number][10]
            else:    
                data=self.all_fit[fit_number][1]
            deconv=self.all_fit[fit_number][5]
            type_fit=self.all_fit[fit_number][8][2]
            exp_no=self.all_fit[fit_number][4]
        else:
            x=self.x
            data=self.data_before_last_Fit
            deconv=self.deconv
            type_fit=self.type_fit
            exp_no=self.exp_no
        ndata, nx = data.shape
        if type_fit == 'Exponential': 
            if deconv:
                self.curve_resultados=data*0.0
                for i in range(nx):
                    self.curve_resultados[:,i]=self.expNGaussDataset(result_params, i,fit_number=fit_number)
            else:
                t0 = result_params['t0_1'].value
                index=np.argmin([abs(i-t0) for i in x])
                self.curve_resultados=0.0*data[index:,:]
                for i in range(nx):
                    self.curve_resultados[:,i]=self.expNDataset(result_params, i,fit_number=fit_number)
        else:
            ksize = exp_no #size of the matrix = no of exponenses = no of species
            kmatrix = np.array([[result_params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
            cinitials = [result_params['c_%i' % (i+1)].value for i in range(ksize)]
            eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
            #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
            eigenmatrix = np.array(vects) 
            coeffs = np.linalg.solve(eigenmatrix, cinitials)
            self.curve_resultados=data*0.0
            for i in range(nx):
                    self.curve_resultados[:,i]=self.expNGaussDatasetTM(result_params, i,[coeffs,eigs,eigenmatrix],fit_number=fit_number)
        return self.curve_resultados

    def parametros(self):
        params = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        ndata, nx = self.data.shape
        for iy in range(nx):
            params.add_many(('y0_' +str(iy+1), 0,True, None, None, None, None),
                            ('t0_'+str(iy+1), 0, False,  np.min(self.x), None, None, None))
            if self.deconv:
                params.add_many(('fwhm_' +str(iy+1), 0.160, False, 0.000001, None, None, None))
                if self.tau_inf is not None:            
                    params.add_many(('yinf_' +str (iy+1), 0.001, True, None, None, None, None))
            
            for i in range (self.exp_no):
                # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                params.add_many(('pre_exp%i_' %(i+1) +str (iy+1), 0.1*10**(-i), True, None, None, None, None),
                                ('tau%i_' %(i+1) +str (iy+1), (i+1)*10, True, 0.00000001, None, None, None))
        return params    
    
    
    def initialParams1(self,number_taus):
        self.exp_no=len(number_taus)
        fit_params=self.parametros()
        self.initial_t0=float(input('introduce value of t0 (default=0)') or '0')
        fit_params['t0_1'].value=self.initial_t0
        if self.deconv:
            self.initial_fwhm=float(input('introduce value of FWHM fro deconvolution (default=0.18)') or '0.18')
            fit_params['fwhm_1'].value=self.initial_fwhm
        for iy in range(2,self.data.shape[1]+1):
            fit_params['t0_%i' % iy].expr='t0_1'
            for i in range (self.exp_no):
                fit_params['tau%i_' %(i+1) +str (iy)].expr='tau%i_1' %(i+1)
                if self.deconv:
                    fit_params['fwhm_%i'% iy].expr='fwhm_1'
        self.initial_taus=[]            
        for i in range (self.exp_no):
            tau=float(input('introduce value of tau%i_1' %(i+1)))
            fit_params['tau%i_1' %(i+1)].value=tau
            self.initial_taus.append(tau)
            fit_params['tau%i_1' %(i+1)].vary=False   
        key ='t0_1'
        if self.deconv:
            ask=input('Has de GVD been corrected? [True]/False (if True t0 will be fix)') or True
            self.initial_GVD_corrected=ask
            if ask != True:
                self.GVD_correction=False
                print('False')
                for iy in range(2,self.data.shape[1]+1):
                    fit_params['t0_%i' % iy].expr=None
                    fit_params['t0_%i' % iy].vary=True
                    self.t0_vary=True
                    
            else:
                answer=input(f'vary {key}? [False]/True') or False
                self.initial_vary_t0=answer
                if answer != False:
                    print('True')
                    fit_params[key].vary=True
                    self.t0_vary=True
                else:
                    print('False')
                    fit_params[key].vary=False
                    self.t0_vary=False
        else:
            answer=input(f'vary {key}? [False]/True') or False
            self.initial_vary_t0=answer
            if answer != False:
                print('True')
                fit_params[key].vary=True
                self.t0_vary=True
            else:
                fit_params[key].vary=False
                self.t0_vary=False
        self.Fit_completed=False
        self.params_initialized=True  
        self.prefit_done=False
        self.initial_params=fit_params
        self.general_report['Sequence of actions'].append('\t--> New parameters initialized')
    
    def initialParams2(self,t0,taus,fwhm=0.16,vary_t0=False,opt_fwhm=False,GVD_corrected=True):
        """function to initialize parameters"""
        self.type_fit='Exponential'
        self.initial_taus=taus
        self.initial_t0=t0
        self.initial_vary_t0=vary_t0
        self.initial_fwhm=fwhm
        self.initial_opt_fwhm=opt_fwhm
        self.initial_GVD_corrected=GVD_corrected
        self.GVD_correction=GVD_corrected
        if type(taus)== float or  type(taus)== int:
            taus=[taus]
        self.exp_no=len(taus)
        fit_params=self.parametros()
        assert len(taus)==self.exp_no
        fit_params['t0_1'].value=t0
        if self.deconv:
            fit_params['fwhm_1'].value=fwhm
            if opt_fwhm:
                fit_params['fwhm_1'].vary=True
        for iy in range(2,self.data.shape[1]+1):
            fit_params['t0_%i' % iy].expr='t0_1'
            for i in range (self.exp_no):
                fit_params['tau%i_' %(i+1) +str (iy)].expr='tau%i_1' %(i+1)
                if self.deconv:
                    fit_params['fwhm_%i'% iy].expr='fwhm_1'
    
        for i in range (self.exp_no):
            fit_params['tau%i_1' %(i+1)].value=taus[i]
        key ='t0_1'
        if self.deconv:
            ask=GVD_corrected
            if ask == False:
                print('GVD corrected: False')
                for iy in range(2,self.data.shape[1]+1):
                    fit_params['t0_%i' % iy].expr=None
                    fit_params['t0_%i' % iy].vary=True
                    self.t0_vary=True
            else:
                answer=vary_t0
                if answer != False:
                    print('vary_t0 True')
                    fit_params[key].vary=True
                    self.t0_vary=True
                else:
                    print('vary_t0 False')
                    fit_params[key].vary=False
                    self.t0_vary=False
        else:
            answer=vary_t0
            if answer != False:
                print('True')
                fit_params[key].vary=True
                self.t0_vary=True
            else:
                fit_params[key].vary=False
                self.t0_vary=False
        self.Fit_completed=False
        self.prefit_done=False
        self.initial_params=fit_params
        self.general_report['Sequence of actions'].append('\t--> New Exponential Fit parameters initialized')
        self.params_initialized=True  
    
    def paramsAlready(self):
       """Allow to create parameters for a new selection of traces without calling the function initialize params"""
       if  self.params_initialized:
           if self.type_fit == 'Exponential':
               if self.Fit_completed:
                   t0 = self.params['t0_1'].value
                   try:
                       fwhm= self.params ['fwhm_1'].value
                   except:
                       fwhm=0.16
                   taus=[self.params['tau%i_1' %(ii+1)].value for ii in range(self.exp_no)] 
                   self.initialParams2(t0,taus,fwhm,self.initial_vary_t0,self.initial_opt_fwhm,self.initial_GVD_corrected)
               else:
                   self.initialParams2(self.initial_t0,self.initial_taus,self.initial_fwhm,self.initial_vary_t0,self.initial_opt_fwhm,self.initial_GVD_corrected)
               del self.general_report['Sequence of actions'][-1]
               self.prefit_done=False
           else:
               t0 = self.params['t0_1'].value
               t0_vary = self.params['t0_1'].vary
               try:
                   fwhm = self.params ['fwhm_1'].value
                   fwhm_vary = self.params ['fwhm_1'].vary
               except:
                   fwhm=0.16
                   fwhm_vary= self.params ['fwhm_1'].value=False
               if self.Fit_completed:
                   for i in range(self.exp_no):
                        for j in range(self.exp_no):
                            self.last_model_params['k_%i%i' % (i+1,j+1)].value=self.params['k_%i%i' % (i+1,j+1)].value
               self.initialParamsModel(self.last_model_params,t0,vary_t0=t0_vary,fwhm=fwhm, vary_fwhm=fwhm_vary)
               del self.general_report['Sequence of actions'][-1]
               self.prefit_done=False 
       else:
           pass
        
    def single_fit(self,params,function,i,extra_params=None):
        """does a fit of a single trace"""
        if extra_params is not None:
            if self.deconv:
                return self.data_before_last_Fit[:, i] - function(params, i, extra_params)
            else:
                t0 = params['t0_%i'%(i+1)].value
                index=np.argmin([abs(i-t0) for i in self.x_before_last_Fit])
                return self.data_before_last_Fit[index:, i] - function(params, i, extra_params)
        else:
            if self.deconv:
                return self.data_before_last_Fit[:, i] - function(params, i)
            else:
                t0 = params['t0_%i'%(i+1)].value
                index=np.argmin([abs(i-t0) for i in self.x_before_last_Fit])
                return self.data_before_last_Fit[index:, i] - function(params, i)
    
    def preFit(self):
        if self.wavelength is not None:
            self.wavelength_before_last_Fit=self.wavelength.copy()
        else:
            self.wavelength_before_last_Fit=None
        #initiate self.data_before_last_Fit copying from self.data which will be used to fit
        #parameters have been created with lenght of self.data
        #this allow to keep after the fit a copy of the data that was fitted
        self.data_before_last_Fit=self.data*1.0
        self.x_before_last_Fit=self.x
        fit_params=self.initial_params.copy()
        ndata, nx = self.data_before_last_Fit.shape
        if self.type_fit == 'Exponential':
            if self.deconv:
                for iy in range(nx,0,-1):#range is decending just for chcking if it will work 
                    single_param=lmfit.Parameters()
                    single_param['y0_%i' %iy]=fit_params['y0_%i' %iy]
                    single_param.add(('t0_%i' %iy), value=fit_params['t0_1'].value,expr=None,vary=self.t0_vary)
                    single_param['fwhm_%i' %iy]=fit_params['fwhm_1']
                    if self.tau_inf is not None:
                        single_param['yinf_%i' %iy]=fit_params['yinf_%i' %iy]
                    for i in range(self.exp_no):
                        single_param.add(('tau%i_' %(i+1) +str (iy)), value=fit_params['tau%i_1' %(i+1)].value,expr=None,vary=False)
                        single_param.add(('pre_exp%i_' %(i+1) +str (iy)),value=fit_params['pre_exp%i_' %(i+1) +str (iy)].value,vary=True)
                    result=lmfit.minimize(self.single_fit,single_param,args=(self.expNGaussDataset, iy-1),nan_policy='propagate')
                    fit_params['y0_%i' %iy]=result.params['y0_%i' %iy]
                    if self.GVD_correction==False:
                        fit_params['t0_%i' %iy]=result.params['t0_%i' %iy]
                    if self.tau_inf is not None:
                        fit_params['yinf_%i' %iy]=result.params['yinf_%i' %iy]        
                    for i in range(self.exp_no):
                        fit_params['pre_exp%i_' %(i+1) +str (iy)]=result.params['pre_exp%i_' %(i+1) +str (iy)]
                    self.params=fit_params
                    self.prefit_done=True
            else:
                for iy in range(nx,0,-1):
#                    print(iy)
                    single_param=lmfit.Parameters()
                    single_param.add(('y0_%i' %iy),value=fit_params['y0_%i' %iy].value,vary=True)
                    single_param.add(('t0_%i' %iy), value=fit_params['t0_1'].value,expr=None,vary=self.t0_vary)
                    for i in range(self.exp_no):
                        single_param.add(('tau%i_' %(i+1) +str (iy)), value=fit_params['tau%i_1' %(i+1)].value,expr=None,vary=False)
                        single_param.add(('pre_exp%i_' %(i+1) +str (iy)),value=fit_params['pre_exp%i_' %(i+1) +str (iy)].value,vary=True)
                    result=lmfit.minimize(self.single_fit,single_param,args=(self.expNDataset, iy-1),nan_policy='propagate')
                    fit_params['y0_%i' %iy].value=result.params['y0_%i' %iy].value
                    #fit_params['t0_%i' %iy].value=result.params['t0_%i' %iy].value
                    for i in range(self.exp_no):
                        fit_params['pre_exp%i_' %(i+1) +str (iy)]=result.params['pre_exp%i_' %(i+1) +str (iy)]
                    self.params=fit_params
        else:
            ksize = self.exp_no #size of the matrix = no of exponenses = no of species
            kmatrix = np.array([[fit_params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
            cinitials = [fit_params['c_%i' % (i+1)].value for i in range(ksize)]
            eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
            #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
            eigenmatrix = np.array(vects) 
            coeffs = np.linalg.solve(eigenmatrix, cinitials) #solve the initial conditions sheet
            #didnt tested but should work, if no then probably minor correction is needed.
            
            for iy in range(nx):
                print(iy)
                single_param=lmfit.Parameters()
                for i in range(self.exp_no):
                    single_param['pre_exp%i_' %(i+1) +str (iy+1)]=fit_params['pre_exp%i_' %(i+1) +str (iy+1)]
                single_param['y0_%i' %(iy+1)]=fit_params['y0_%i' %(iy+1)]
                single_param.add(('t0_%i' %(iy+1)), value=fit_params['t0_1'].value,expr=None,vary=self.t0_vary)
                if self.deconv:
                    single_param.add(('fwhm_%i' %(iy+1)), value=fit_params['fwhm_1'].value,expr=None,vary=fit_params['fwhm_1'].vary)
                result=lmfit.minimize(self.single_fit,single_param,args=(self.expNGaussDatasetTM, iy,[coeffs,eigs,eigenmatrix]),nan_policy='propagate')    
                if self.GVD_correction==False and self.deconv:
                    fit_params['t0_%i' %(iy+1)]=result.params['t0_%i' %(iy+1)]
                for i in range(self.exp_no):
                    fit_params['pre_exp%i_' %(i+1) +str (iy+1)]=result.params['pre_exp%i_' %(i+1) +str (iy+1)]
                self.params=fit_params
        self.prefit_done=True                            
    
    def expNGaussDatasetTM(self,params, i,cons_eigen,fit_number=None):
        if fit_number is not None:
            x=self.all_fit[fit_number][0]
            exp_no=self.all_fit[fit_number][4]
            deconv=self.all_fit[fit_number][5]
        else:
            exp_no=self.exp_no
            x=self.x
            deconv = self.deconv
        y0 = params['y0_%i' % (i+1)].value
        t0 = params['t0_%i' % (i+1)].value
        pre_exp = [params ['pre_exp%i_' % (ii+1)+str (i+1)].value for ii in range(exp_no)]
        coeffs,eigs,eigenmatrix = cons_eigen[0],cons_eigen[1],cons_eigen[2]
        if deconv:
            fwhm = params ['fwhm_%i' % (i+1)].value
            expvects = [coeffs[val]*self.expGaussConv(x-t0,-eigs[val],fwhm/2.35482) for val in range(len(eigs))]
        else:
            t0 = params['t0_1'].value
            index=np.argmin([abs(i-t0) for i in x])
            expvects = [coeffs[val]*self.exp1(x[index:]-t0,-eigs[val]) for val in range(len(eigs))]
        concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
        return y0+sum([pre_exp[iii]*concentrations[iii] for iii in range(exp_no)])                
        
    
    def finalFit(self,vary_taus=True,maxfev=None,time_constraint=False,save_fit=True,apply_weights=False):
        if type(vary_taus)== bool:
            vary_taus=[vary_taus for i in range(self.exp_no)]
        self.Fit_completed=False
        if self.prefit_done==False:
            self.preFit()
        if self.wavelength is not None:
            self.wavelength_before_last_Fit=self.wavelength.copy()
        else:
            self.wavelength_before_last_Fit=None
        fit_condition=[maxfev,time_constraint,self.type_fit]#self.type_fit is important to know if we are doing an expoential or taget fit
        self.data_before_last_Fit=self.data.copy()#important for bootstrapping
        self.x_before_last_Fit=self.x
        self.initial_vary_taus=vary_taus#important for bootstrapping
        self.initail_maxfev=maxfev#important for bootstrapping
        self.initial_time_constraint=time_constraint#important for bootstrapping
        if type(vary_taus)== bool:
            vary_taus=[vary_taus]
        fit_params=self.params
        ndata, nx = self.data_before_last_Fit.shape
        if self.type_fit == 'Exponential':#in case of exponential fit verify the consistency
            if len(vary_taus) == self.exp_no:
                for i in range (self.exp_no):
                    key='tau%i_1' %(i+1)
                    fit_params[key].vary=vary_taus[i]
            else:
                for i in range (self.exp_no):
                    key='tau%i_1' %(i+1)
                    answer=input(f'vary {key} the value is {fit_params[key].value}? [True]/False') or True
                    if answer != True:
                        print('False')
                        fit_params[key].vary=False
                    else:
                        print('True')
                        fit_params[key].vary=True
            if time_constraint:
                for i in range (self.exp_no):
                    if i == 0:
                        if self.deconv:
                            fit_params['tau%i_1' %(i+1)].min=fit_params['fwhm_1'].value
                        else:
                            fit_params['tau%i_1' %(i+1)].min=fit_params['t0_1'].value
                    else:
                      fit_params['tau%i_1' %(i+1)].min=fit_params['tau%i_1' %(i)].value 
                  
        if self.deconv:
            names=['t0_1','fwhm_1']+['tau%i_1' %(i+1) for i in range(self.exp_no)]
            deconvolution=f'deconvolved with a gaussian, tau_inf={self.tau_inf}'
        else:
            names=['t0_1']+['tau%i_1' %(i+1) for i in range(self.exp_no)]
            deconvolution='No deconvolved'
        if apply_weights and len(self.weights['vector'])==len(self.x):
            self.weights['apply']=True
            fit_condition.append(self.weights)
            weight_rep=f'weights in range: {self.weights["range"][0]}-{self.weights["range"][1]} {self.time_unit}, type: {self.weights["type"]}, value {self.weights["value"]}\n'
        else:
            fit_condition.append('no weights')
            weight_rep=''
        if maxfev!=None:
            self.resultados=self.minimize(params=fit_params,maxfev=maxfev)
        else:
            self.resultados=self.minimize(params=fit_params)
        self.number_it = 0
        self.Fit_completed=True    
        if self.type_fit != 'Exponential':
            for i in range(self.exp_no):
                if(self.resultados.params['k_%i%i' % (i+1,i+1)].value != 0):
                    self.resultados.params['tau%i_1' % (i+1)].value=-1/self.resultados.params['k_%i%i' % (i+1,i+1)].value
                else:
                    self.resultados.params['tau%i_1' % (i+1)].value=np.inf
        self.params=deepcopy(self.resultados.params)
        if type(fit_condition[3]) == dict:
            self.weights['apply']=False
        if self.SVD_fit:
            self.afterSVDFit(fit_condition,names,deconvolution,save_fit,weight_rep)
            save_fit=False
        
        if save_fit:
            self.fit_number += 1
            self.all_fit[self.fit_number]=(self.x_before_last_Fit,self.data_before_last_Fit,self.wavelength_before_last_Fit,self.resultados,
                        self.exp_no,self.deconv,self.tau_inf,self.derivative_space,fit_condition,self.SVD_fit)
            print_resultados='\t\t'+',\n\t\t'.join([f'{name.split("_")[0]}:\t{round(self.resultados.params[name].value,4)}\t{self.resultados.params[name].vary}' for name in names])
            self.general_report['Fits done'][f'Fit number {self.fit_number}']=f'Global {self.type_fit} Fit\n\t---------------\n\tFitted {self.data_before_last_Fit.shape[1]} traces, average {self.average_trace}, with {self.exp_no} exponenitial,\n\t{deconvolution},\n\tresults:\tname\tvalue\toptimized\n{print_resultados}\n\t{weight_rep}'
            self.general_report['Sequence of actions'].append(f'\t--> Global {self.type_fit} Fit {self.fit_number} completed') 

    def afterSVDFit(self,fit_condition,names,deconvolution,save_fit,weight_rep):
        self.fit_number += 1
        resultados=self.resultados
        data_SVD=self.data*1.0
        initial_params=self.initial_params.copy()
        self.data=self.data_before_first_selection*1.0
        self.paramsAlready()
        self.preFit()
        self.Fit_completed==True
        self.data=data_SVD*1.0
        if save_fit:
            print('true')
            self.all_fit[self.fit_number]=(self.x_before_last_Fit,self.data_before_last_Fit,self.wavelength_before_first_selection,self.resultados,
                        self.exp_no,self.deconv,self.tau_inf,self.derivative_space,fit_condition,self.SVD_fit,data_SVD,self.params)
            print_resultados='\t\t'+',\n\t\t'.join([f'{name.split("_")[0]}:\t{round(self.resultados.params[name].value,4)}\t{self.resultados.params[name].vary}' for name in names])
            self.general_report['Fits done'][f'Fit number {self.fit_number}']=f'Singular vector {self.type_fit} Fit\n\t---------------\n\tFitted {data_SVD.shape[1]} Singular vectors with {self.exp_no} exponenitial,\n\t{deconvolution},\n\tresults:\tname\tvalue\toptimized\n{print_resultados}\n\t{weight_rep}'
            self.general_report['Sequence of actions'].append(f'\t--> Singular vector {self.type_fit} Fit {self.fit_number} completed') 
        self.data_before_last_Fit=data_SVD*1.0
        self.params=deepcopy(resultados.params)
        self.initial_params=initial_params.copy()
        
        print('finish')
        
    
    def plotFit(self,size=14,save=False,fit_number=None,selection=None,plot_residues=True):      
        if fit_number is not None:
            x=self.all_fit[fit_number][0]
            #verify type of fit is: either fit to Singular vectors or global fit to traces
            SVD_fit=self.all_fit[fit_number][9] 
            if SVD_fit:
                data=self.all_fit[fit_number][10]
            else:
                data=self.all_fit[fit_number][1]
            wavelenght=self.all_fit[fit_number][2]
            params=self.all_fit[fit_number][3].params
            deconv=self.all_fit[fit_number][5]
        else:
            data=self.data_before_last_Fit
            x=self.x_before_last_Fit
            wavelenght=self.wavelength_before_last_Fit
            params=self.params
            deconv=self.deconv
            SVD_fit=self.SVD_fit
        if wavelenght is None:
            wavelenght=np.array([i for i in range(len(data[1]))])
        if SVD_fit:
            selection=None
        if selection is None:
            puntos=[i for i in range(data.shape[1])]
        else:
            puntos=[min(range(len(wavelenght)), key=lambda i: abs(wavelenght[i]-num)) for num in selection]
        if len(puntos)<=10:
            if SVD_fit:
                legenda=['_' for i in range(data.shape[1]+1)]+['left SV %i' %i for i in range(1,data.shape[1]+1)]
            elif wavelenght is not None:
                legenda=['_' for i in range(len(puntos)+1)]+[f'{round(wavelenght[i])} nm' for i in puntos]
            else:
                legenda=['_' for i in range(len(puntos)+1)]+[f'curve {i}' for i in  range(data.shape[1])]
        xlabel='Time ('+self.time_unit+')'
        if plot_residues==False:
            fig, ax = plt.subplots(figsize=(8,6))
            ax=['_',ax]
        else:
            fig, ax = plt.subplots(2, 1,sharex=True, figsize=(8,6), gridspec_kw={'height_ratios': [1, 5]})
        fittes=self.results(params,fit_number=fit_number)
        if deconv:
            self.residues= 0.0*data[:]
        else:
             t0 = params['t0_1'].value
             index=np.argmin([abs(i-t0) for i in x])
             self.residues= 0.0*data[index:,:]
        plt.axhline(linewidth=1,linestyle='--', color='k')
        alpha,s=0.80,8
        if self.alpha['activate']:
            alpha,s=self.alpha['value'],self.alpha['size']
        for i in puntos:
            if deconv:
                if plot_residues:
                    self.residues[:,i]=data[:,i]-fittes[:,i]
                    ax[0].scatter(x,self.residues[:,i], marker='o',alpha=alpha,s=s)
                ax[1].plot(x, fittes[:,i], '-',color='r',alpha=0.5,lw=1.5)
            else:
                if plot_residues:
                    self.residues[:,i]=data[index:,i]-fittes[:,i]
                    ax[0].scatter(x[index:],self.residues[:,i], marker='o',alpha=alpha,s=s)
                ax[1].plot(x[index:], fittes[:,i], '-',color='r',alpha=0.5,lw=1.5)
            ax[1].scatter(x,data[:,i], marker='o',alpha=alpha,s=s)
            if len(puntos)<=10:
                ax[1].legend(legenda,loc='best',ncol=1 if SVD_fit else 2)
        if plot_residues:
            ax[0].ticklabel_format(style='sci',axis='y')
            ax[0].minorticks_on()
            ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            ax[0].set_ylim(np.min(self.residues)-abs(np.min(self.residues)*0.1),np.max(self.residues)+np.max(self.residues)*0.1)
            ax[0].set_ylabel('Residues',size=size)
        ax[1].ticklabel_format(style='sci',axis='y')
        ax[1].minorticks_on()
        ax[1].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
        ax[1].set_ylim(np.min(data)-abs(np.min(data)*0.1),np.max(data)+np.max(data)*0.1)
        #f.tight_layout()
        ax[1].set_xlabel(xlabel, size=size)
        plt.xlim(x[0]-x[-1]/50,x[-1]+x[-1]/50)
        ax[1].set_ylabel(r'$\Delta$A',size=size)
        plt.subplots_adjust(left=0.145,right=0.95)
        if save:
            if self.save['name']=='':
                self.save['name']=f'Fit with {self.exp_no} exp'
                if selection is not None:
                    self.save['name'] ='selection of traces' + self.save['name']
            fig.savefig(self.save['path']+self.save['name']+'.'+self.save['format'], dpi=self.save['dpi'])
            self.save['name']=''
        return fig, ax
            
    def DAS(self,fit_number=None):
        if fit_number is not None:
            #verify type of fit is: either fit to Singular vectors or global fit to traces
            if self.all_fit[fit_number][9]:#verify type of fitaither SVD or global fit
                result_params=self.all_fit[fit_number][11]
            else:    
                result_params=self.all_fit[fit_number][3].params
            data=self.all_fit[fit_number][1]
            exp_no=self.all_fit[fit_number][4]
            deconv=self.all_fit[fit_number][5]
            tau_inf=self.all_fit[fit_number][6]
        else:
            result_params=self.params
            data=self.data_before_last_Fit
            exp_no=self.exp_no
            deconv=self.deconv
            tau_inf=self.tau_inf
        values=[[result_params['pre_exp%i_' % (ii+1)+str(i+1)].value for i in range(data.shape[1])] for ii in range(exp_no)]
        if deconv and tau_inf is not None:
            values.append([result_params['yinf_'+str(i+1)].value for i in range(data.shape[1])])
        return np.array(values)
    
    def plotDAS(self,number='all',fit_number=None,size=14,save=False,precision=2,plot_derivat_DAS=False):
        '''REtunrs the Decay Asociated Spectra of the FIt
        Parameters: 
            number --> 'all' or a list conataining the decay of the species you want, plus -1 if you want the tau inf if existing
                        eg1.: if you want the decay of the second and third, [2,3] // eg2.:  fist third and inf, [2,3,-1]
                            
        ''' 
        if fit_number is not None:
            #verify type of fit is: either fit to Singular vectors or global fit to traces
            if self.all_fit[fit_number][9]:
                params=self.all_fit[fit_number][11]
            else:
                params=self.all_fit[fit_number][3].params
            exp_no=self.all_fit[fit_number][4]
            deconv=self.all_fit[fit_number][5]
            wavelenght=self.all_fit[fit_number][2]
            tau_inf=self.all_fit[fit_number][6]
            derivative_space=self.all_fit[fit_number][7]
            type_fit=self.all_fit[fit_number][8][2]#check for type of fit done target or exponential
        else:
            exp_no=self.exp_no
            deconv=self.deconv
            wavelenght=self.wavelength_before_last_Fit
            tau_inf=self.tau_inf
            derivative_space=self.derivative_space
            type_fit=self.type_fit
            params=self.params
        if number is not 'all':
            assert type(number)==list, 'NUmber should be "all" or a list containing the desired species if tau inf include -1 in the list'
            das=self.DAS(fit_number=fit_number)
            posible=[i+1 for i in range(exp_no)]
            if tau_inf is not None:
                posible.append(-1)
            wanted=[ii for ii,i in enumerate(posible) if i in number ]    
            das=das[wanted,:]
        else:
            das=self.DAS(fit_number=fit_number)
        if self.wavelength_unit=='cm-1':
            xlabel='Wavenumber (cm$^{-1}$)'
        else:
            xlabel='Wavelength (nm)'
        legenda=[]
        for i in range(exp_no):
            tau=params['tau%i_1' % (i+1)].value
            if tau<0.09:
                tau*=self.factor_low
                legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,precision))+' '+self.time_unit_low)
            elif tau>999:
                if tau>1E12:
                    legenda.append(r'$\tau$ = inf')
                else:    
                    tau/=self.factor_high
                    legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,precision))+' '+self.time_unit_high)
            else:
                legenda.append(rf'$\tau {i+1}$ = '+'{:.2f}'.format(round(tau,precision))+' '+self.time_unit)
        if deconv and type_fit == 'Exponential':
            if tau_inf is None:
                pass
            elif tau_inf != 1E+12:
                legenda.append(r'$\tau$ = {:.2f}'.format(round(self.tau_inf/self.factor_high,precision))+' '+self.time_unit_high)  
            else:
                legenda.append(r'$\tau$ = inf')
        if number is not 'all':
#            constants=' '.join([str(i.split('=')[1]) for i in legenda[:number]])
#            print(f'WARNING! time constants of value {constants} has been used to fit')
            legenda=[legenda[i] for i in wanted]
        if type(derivative_space)==dict and plot_derivat_DAS:
            das_deriv=das
            for i in range(derivative_space['deriv']):
                das_deriv=np.array([integral.cumtrapz(das_deriv[i,:],wavelenght,initial=das_deriv[i,0]) for i in range(len(das_deriv))])
            fig,ax =  plt.subplots(1,2,figsize=(18,6))
            for i in range(das.shape[0]):
                ax[0].plot(wavelenght,das[i,:],label=legenda[i])
                ax[1].plot(wavelenght,das_deriv[i,:],label=legenda[i])
            ax[0].set_xlim(wavelenght[0],wavelenght[-1])
            leg=ax[0].legend(prop={'size': size})
            leg.set_zorder(np.inf)
            ax[0].axhline(linewidth=1,linestyle='--', color='k')
            ax[0].minorticks_on()
            ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            ax[0].set_xlabel(xlabel,size=size)
            ax[0].set_ylabel('$\Delta$A',size=size)
            ax[1].set_xlim(wavelenght[0],wavelenght[-1])
            leg=ax[1].legend(prop={'size': size})
            leg.set_zorder(np.inf)
            ax[1].axhline(linewidth=1,linestyle='--', color='k')
            ax[1].minorticks_on()
            ax[1].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            ax[1].set_xlabel(xlabel,size=size)
            ax[1].set_ylabel('$\Delta$A',size=size)
        
        else:   
            if type(derivative_space)==dict:
                das=np.array([integral.cumtrapz(das[i,:],wavelenght,initial=0) for i in range(len(das))])
            fig,ax = plt.subplots(1,figsize=(11,6))
            for i in range(das.shape[0]):
                ax.plot(wavelenght,das[i,:],label=legenda[i])
                plt.xlim(wavelenght[0],wavelenght[-1])
            leg=ax.legend(prop={'size': size})
            leg.set_zorder(np.inf)
            plt.axhline(linewidth=1,linestyle='--', color='k')
            #plt.ticklabel_format(style='sci',scilimits=(-0.1,0.1),axis='y')
            ax.minorticks_on()
            ax.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            plt.xlabel(xlabel,size=size)
            plt.ylabel('$\Delta$A',size=size)
            if self.excitation is not None and self.excitation>np.min(wavelenght) and self.excitation<np.max(wavelenght) or self.inner_cut_done is not None:
                if self.excitation is not None:
                    ymin, ymax = ax.get_ylim()
                    ymin = ymin-ymin*0.05
                    ymax = ymax - ymax*0.05
                    index=np.argmin([abs(self.excitation-i) for i in  wavelenght])
                if self.inner_cut_done is not None:
                    mini=np.argmin([abs(self.inner_cut_done[0]-i) for i in wavelenght])
                    maxi=np.argmin([abs(self.inner_cut_done[1]-i) for i in wavelenght])
                    initial=wavelenght[mini]
                    final=wavelenght[maxi]
                elif wavelenght[index]>self.excitation:
                    initial=wavelenght[index-1]
                    final=wavelenght[index]
                else:
                    initial=wavelenght[index]
                    final=wavelenght[index+1]
                rect = Rectangle([initial-1,ymin],width =final-initial+2, 
                                           height=abs(ymax)+abs(ymin),fill=True, color='white',zorder=5+1)
                ax.add_patch(rect)
        if save:
            if self.save['name']=='':
                if plot_derivat_DAS and type(self.derivative_space) is dict:
                    self.save['name']=f'DAS {self.exp_no} exp and derivate space'
                else:    
                    self.save['name']=f'DAS {self.exp_no} exp'
            fig.savefig(self.save['path']+self.save['name']+'.'+self.save['format'], dpi=self.save['dpi'])
            self.save['name']=''   
        return fig, ax
     
    def plotRawData(self,size=14,auto=False,select=False,select_number=-1,save=False):
        values=[i for i in range(self.data.shape[1])]
        data_changed=False
        if auto:
            if self.data_before_first_selection is not None:
                previous_data=self.data
                previous_wave=self.wavelength
                self.data=self.data_before_first_selection
                self.wavelength=self.wavelength_before_first_selection
                data_changed=True
            values=[i for i in range(len(self.wavelength))[::round(len(self.wavelength)/11)]]
            values=values[1:]
        if self.data.shape[1]<=10 or auto:
            if self.SVD_fit:
                legenda=['left SV %i' %i for i in range(1,self.data.shape[1]+1)]
            elif self.wavelength is not None:
                if auto:
                    legenda=[f'{round(i)} nm' for i in self.wavelength[values]]
                else:
                    legenda=[f'{round(i)} nm' for i in self.wavelength]
            else:
                legenda=[f'curve {i}' for i in  range(len(self.data.shape[1]))]
        xlabel='Time ('+self.time_unit+')'
        fig,a0 = plt.subplots(1,figsize=(11,6))
        alpha,s=0.60,8
        if self.alpha['activate']:
            alpha,s=self.alpha['value'],self.alpha['size']
        for i in values:
            a0.plot(self.x,self.data[:,i],marker='o',alpha=alpha,ms=4,ls='')
        a0.set_ylim(np.min(self.data)-abs(np.min(self.data)*0.1),np.max(self.data)+np.max(self.data)*0.1)
        a0.set_xlim(self.x[0]-self.x[-1]/50,self.x[-1]+self.x[-1]/50)
        a0.axhline(linewidth=1,linestyle='--', color='k')
        if self.data.shape[1]<=10 or auto:
            a0.legend(legenda,loc='best',ncol=2)
        a0.ticklabel_format(style='sci',axis='y')
        #f.tight_layout()
        a0.set_ylabel('$\Delta$A',size=size)
        a0.set_xlabel(xlabel,size=size)
        a0.minorticks_on()
        a0.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
        if save:
            if self.save['name']=='':
                self.save['name']='raw data'
            fig.savefig(self.save['path']+self.save['name']+'.'+self.save['format'], dpi=self.save['dpi'])
            self.save['name']=''
        if select:
            cursor = SnaptoCursor(a0, self.x,self.x*0.0)
            plt.connect('axes_enter_event', cursor.onEnterAxes)
            plt.connect('axes_leave_event', cursor.onLeaveAxes)
            plt.connect('motion_notify_event', cursor.mouseMove)
            points=plt.ginput(n=select_number,timeout=30,show_clicks=True)
            self.selected_time=sorted([(pd.Series(self.x)-points[i][0]).abs().sort_values().index[0] for i in range(len(points))])
        if data_changed:
            self.data=previous_data
            self.wavelength=previous_wave
        return fig,a0

    def plotCalculatedSpectra(self, times,n_points,cmap=None,ncol=1, size=14,legend=True, plot_raw=True,save=False,):
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
        if times[0]=='auto':
            if len(times)==1:
                times=self.getAutoPoints()
            elif len(times)==2:
                times=self.getAutoPoints(times[1])
            elif len(times)==3:
                times=self.getAutoPoints(times[1],times[2])   
            else:
                print('if first element is "auto" then spectra will be auto plotted \n \
                      then the list can be only   ["auto"] or:\n\
                      ["auto", number of spectra(optional; int),  wavelenght to select spectra(optional; int)],\n \
                      if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",15,wavelenght] 15 spectra will be plotted equally spaced at the selected wavelenght')
                
        elif times is 'all':
            times=self.x
            legend=False
        elif times is 'auto': 
            times=self.getAutoPoints()
        if self.wavelength_unit=='cm-1':
            xlabel='Wavenumber (cm$^{-1}$)'
        else:
            xlabel=f'Wavelength ({self.wavelength_unit})'
        tiempo=pd.Series(self.x)
        real_times=[tiempo[(tiempo-i).abs().sort_values().index[0]] for i in times]
        legenda=['{:.2f}'.format(i*self.factor_low) + ' '+ self.time_unit_low if i<0.09
                 else '{:.2f}'.format(i) + ' '+ self.time_unit if i<999
                 else '{:.2f}'.format(i/self.factor_high) + ' '+ self.time_unit_high for i in real_times]
        a=np.linspace(0,1,len(times))
        c=plt.cm.ScalarMappable(norm=[0,1],cmap=cmap)
        colors=c.to_rgba(a,norm=False)
        times=list(times)
        cols=[str(i) for i in self.wavelength]
        data=pd.DataFrame(columns=cols,data=self.results(self.resultados.params))
        self.recompose_spec=pd.DataFrame(columns=([self.time_unit]+cols))
        if plot_raw:
            data_raw=pd.DataFrame(columns=cols,data=self.data)
            self.spectra=pd.DataFrame(columns=([self.time_unit]+cols))
            self.recompose_spec=pd.DataFrame(columns=([self.time_unit]+cols))
            fig, a1 = plt.subplots(1,2,figsize=(18,6))
            for i in range(len(times)):
                index=(tiempo-times[i]).abs().sort_values().index[0]
                self.spectra.loc[i+1]=[tiempo[index]]+list((data_raw.iloc[index-n_points:index+n_points,:]).mean())
                index=(tiempo-times[i]).abs().sort_values().index[0]
                self.recompose_spec.loc[i+1]=[tiempo[index]]+list((data.iloc[index-n_points:index+n_points,:]).mean())
                a1[0].plot(self.wavelength,self.spectra.loc[i+1,cols[0]:],c=colors[i])
                a1[1].plot(self.wavelength,self.recompose_spec.loc[i+1,cols[0]:],c=colors[i])
            a1[0].set_xlim(self.wavelength[0],self.wavelength[-1])
            a1[0].ticklabel_format(style='sci',axis='y')
            if legend:
                a1[0].legend(legenda,loc='best',ncol=ncol,prop={'size': size})
            a1[0].set_ylabel('$\Delta$A',size=size)
            a1[0].set_xlabel(xlabel,size=size)
            a1[0].minorticks_on()
            a1[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            a1[0].ticklabel_format(style='sci',axis='y')
            a1[0].axhline(linewidth=1,linestyle='--', color='k')
            a1[0].set_title('Raw spectra')
            a1[1].set_xlim(self.wavelength[0],self.wavelength[-1])
            a1[1].ticklabel_format(style='sci',axis='y')
            if legend:
                a1[1].legend(legenda,loc='best',ncol=ncol,prop={'size': size})
            else:
                cnorm = Normalize(vmin=times[0],vmax=times[-1])
                cpickmap = plt.cm.ScalarMappable(norm=cnorm,cmap=cmap)
                cpickmap.set_array([])
            a1[1].set_xlabel(xlabel,size=size)
            a1[1].minorticks_on()
            a1[1].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            a1[1].ticklabel_format(style='sci',axis='y')
            a1[1].axhline(linewidth=1,linestyle='--', color='k')
            a1[1].set_title('Calculated spectra')
            fig.tight_layout()
            
        else:
            fig,a1 = plt.subplots(1,figsize=(11,6))
            for i in range(len(times)):
                index=(tiempo-times[i]).abs().sort_values().index[0]
                self.recompose_spec.loc[i+1]=[tiempo[index]]+list((data.iloc[index-n_points:index+n_points,:]).mean())
                a1.plot(self.wavelength,self.recompose_spec.loc[i+1,cols[0]:],c=colors[i])
            if legend:
                plt.legend(legenda,loc='best',ncol=ncol,prop={'size': size})
            else:
                cnorm = Normalize(vmin=times[0],vmax=times[-1])
                cpickmap = plt.cm.ScalarMappable(norm=cnorm,cmap=cmap)
                cpickmap.set_array([])
            plt.colorbar(cpickmap).set_label(label='Time ('+self.time_unit+')',size=15)
            plt.xlim(self.wavelength[0],self.wavelength[-1])
            a1.ticklabel_format(style='sci',axis='y')
            plt.legend(legenda,loc='best',prop={'size': size})
            a1.set_ylabel('$\Delta$A',size=size)
            a1.set_xlabel(xlabel,size=size)
            a1.minorticks_on()
            a1.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
        if save:
            if self.save['name']=='':
                    self.save['name']=f'calculated spectra with {self.exp_no} exp from {times[0]}-{times[-1]} µs'
            fig.savefig(self.save['path']+self.save['name']+'.'+self.save['format'], dpi=self.save['dpi'])
            self.save['name']=''
        return fig, a1
        
    def plot3D(self,cmap=None):
        if cmap is None:
            cmap=self.color_map
        X=self.x
        if self.data_before_first_selection is not None:
            Z=self.data_before_first_selection.transpose()
            Y=self.wavelength_before_first_selection
        else:
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

        # Customize the z axis.
        ax.set_zlim(np.min(Z), np.max(Z))
#        ax.zaxis.set_major_locator(LinearLocator(10))
#        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel(f'Time ({self.time_unit})')
        ax.set_ylabel(xlabel)
        ax.set_zlabel('$\Delta$A')
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        return fig,ax
    
    def plotSpectra(self, times='all',rango=None,n_points=0,excitation_width=None,
                    cmap=None,ncol=1,size=14,save=False,legend=True, select=False,select_number=-1,include_rango_max=True): 
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
        if self.data_before_first_selection is not None:
            data=self.data_before_first_selection
            wavelength=self.wavelength_before_first_selection
        else:
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
        if data.shape[0]>400 and times=='all' and save==False:
            times='auto'
            print('More than 400 spectra cannot be plotted or your computer risk of running out of memory')
        
        if select:
            if data.shape[0]>250:
                times='auto'
            else:
                times='all'
        if times[0]=='auto':
            if len(times)==1:
                times=self.getAutoPoints(rango=rango,include_rango_max = include_rango_max)
            elif len(times)==2:
                times=self.getAutoPoints(spectra=times[1],rango=rango,include_rango_max = include_rango_max)
            elif len(times)==3:
                times=self.getAutoPoints(times[1],times[2],rango=rango,include_rango_max = include_rango_max)   
            else:
                print('if first element is "auto" then spectra will be auto plotted \n \
                      then the list can be only   ["auto"] or:\n\
                      ["auto", number of spectra(optional; int),  wavelenght to select spectra(optional; int)],\n \
                      if only ["auto"] 8 spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",n_number_spec] n_number spectra will be plotted equally spaced at the maximum for all wavelengths\n \
                      if ["auto",15,wavelenght] 15 spectra will be plotted equally spaced at the selected wavelenght')
        
        elif times is 'all':
            times=x
            legend=False
            if save==True and self.save['name']=='':
                self.save['name']=f'all spectra'
        elif times is 'auto': 
            times=self.getAutoPoints(rango=rango,include_rango_max=include_rango_max)
        if self.wavelength_unit=='cm-1':
            xlabel='Wavenumber (cm$^{-1}$)'
        else:
            xlabel=f'Wavelength ({self.wavelength_unit})'
        if wavelength is None:
            wavelength=np.array([i for i in range(len(data[1]))])
            xlabel='pixel'
        tiempo=pd.Series(x)
        times=sorted(list(set(times)))
        if times is 'all':
            real_times=x
        else:
            real_times=[tiempo[(tiempo-i).abs().sort_values().index[0]] for i in times]
        legenda=['{:.2f}'.format(i*self.factor_low) + ' '+ self.time_unit_low if abs(i)<0.09
                 else '{:.2f}'.format(i) + ' '+ self.time_unit if i<999
                 else '{:.2f}'.format(i/self.factor_high) + ' '+ self.time_unit_high for i in real_times]
        a=np.linspace(0,1,len(times))
        c=plt.cm.ScalarMappable(norm=[0,1],cmap=cmap)
        colors=c.to_rgba(a,norm=False)
        cols=[str(i) for i in wavelength]
        data=pd.DataFrame(columns=cols,data=data)
        fig,a1 = plt.subplots(1,figsize=(11,6))
        self.spectra=pd.DataFrame(columns=([self.time_unit]+cols))
        for i in range(len(times)):
            index=(tiempo-times[i]).abs().sort_values().index[0]
            if n_points==0:
                self.spectra.loc[i+1]=[tiempo[index]]+list(data.iloc[index,:])
            else:
                self.spectra.loc[i+1]=[tiempo[index]]+list((data.iloc[index-n_points:index+n_points,:]).mean())
            a1.plot(wavelength,self.spectra.loc[i+1,cols[0]:],c=colors[i],label=legenda[i])
        plt.xlim(wavelength[0],wavelength[-1])
        a1.ticklabel_format(style='sci',axis='y')
        if legend:
            leg=plt.legend(loc='best',ncol=ncol,prop={'size': size})
            leg.set_zorder(np.inf)
        else:
            cnorm = Normalize(vmin=times[0],vmax=times[-1])
            cpickmap = plt.cm.ScalarMappable(norm=cnorm,cmap=cmap)
            cpickmap.set_array([])
            plt.colorbar(cpickmap).set_label(label='Time ('+self.time_unit+')',size=15)
        a1.axhline(linewidth=1,linestyle='--', color='k')
        a1.set_ylabel('$\Delta$A',size=size)
        a1.set_xlabel(xlabel,size=size)
        a1.minorticks_on()
        a1.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size,zorder=len(times)+1000)
        if self.inner_cut_done is not None:
            ymin, ymax = a1.get_ylim()
            ymin = ymin-ymin*0.05
            ymax = ymax - ymax*0.05
            mini=np.argmin([abs(self.inner_cut_done[0]-i) for i in wavelength])
            maxi=np.argmin([abs(self.inner_cut_done[1]-i) for i in wavelength])
            rect = Rectangle([wavelength[mini]-1,ymin],width =wavelength[maxi]-wavelength[mini]+2, 
                                       height=abs(ymax)+abs(ymin),fill=True, color='white',zorder=len(times))
            a1.add_patch(rect)
        elif self.excitation is not None and self.excitation>np.min(wavelength) and self.excitation<np.max(wavelength):
            if excitation_width is None:
                excitation_width=self.excitation_width
#            lista=[i for i,ii in enumerate(a.columns) if a[ii].all()==0]
            ymin, ymax = a1.get_ylim()
            ymin = ymin-ymin*0.05
            ymax = ymax - ymax*0.05
            index=np.argmin([abs(self.excitation-i) for i in wavelength])
            rect = Rectangle([self.excitation-11,ymin],width =excitation_width*2, 
                                       height=abs(ymax)+abs(ymin),fill=True, color='white',zorder=len(times))
            a1.add_patch(rect)
        if select:
            cursor = SnaptoCursor(a1, wavelength,wavelength*0.0)
            plt.connect('axes_enter_event', cursor.onEnterAxes)
            plt.connect('axes_leave_event', cursor.onLeaveAxes)
            plt.connect('motion_notify_event', cursor.mouseMove)
            points=plt.ginput(n=select_number,timeout=30,show_clicks=True)
            self.selected_traces=sorted([(pd.Series(wavelength)-points[i][0]).abs().sort_values().index[0] for i in range(len(points))])
        if save:
            if self.save['name']=='':
                if times=='auto':
                    self.save['name']=f'auto plotted spectra'
                elif times=='all':
                    self.save['name']=f'all  spectra'
                else:       
                    self.save['name']=f'selected spectra from {times[0]}-{times[-1]} {self.time_unit}'
            fig.savefig(self.save['path']+self.save['name']+'.'+self.save['format'], dpi=self.save['dpi'])
            self.save['name']=''
        return fig, a1
    
    def delPoints(self,points,dimension='time'):
        assert dimension == 'time' or dimension=='wavelength'
        data_changed=False
        if self.data_before_first_selection is not None:
            previous_data=self.data
            previous_wave=self.wavelength
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
            data_changed=True
        if self.data_before_del_point is None:
            self.preprocess_del_point=self.general_report['Preprocessing']
            self.data_before_del_point=self.data
            self.wavelength_before_del_point=self.wavelength
            self.x_before_del_point=self.x
        if type(points) is int or type(points) is float:
            points=[points]
        if dimension=='wavelength':
            if self.wavelength is None:
                wavelength=np.array([i for i in range(len(self.data[1]))])
            else:
                wavelength=self.wavelength
            index=[np.argmin(abs(wavelength-i)) for i in points]
            self.data=np.delete(self.data,index,axis=1)
            self.wavelength=np.delete(self.wavelength,index)
            if data_changed:
                previous_wave=np.delete(previous_wave,index)
                previous_data=np.delete(previous_data,index,axis=1)
            for i in points:
                self.general_report['Preprocessing']['Deleted wavelength points'].append(float(i))
            self.general_report['Data Shape']['Actual number of traces']=self.data.shape[0]
        else:
            index=[np.argmin(abs(self.x-i)) for i in points]
            self.data=np.delete(self.data,index,axis=0)
            self.x=np.delete(self.x,index)
            for i in points:
                self.general_report['Preprocessing']['Deleted time points'].append(float(i))
            self.general_report['Data Shape']['Actual time points']=self.data.shape[0]    
            if data_changed:
                previous_data=np.delete(previous_data,index,axis=0)
        self.general_report['Sequence of actions'].append(f'\t--> {dimension} points deleted')
        if data_changed:
            self.data_before_first_selection=self.data
            self.wavelength_before_first_selection=self.wavelength
            self.data=previous_data
            self.wavelength=previous_wave
    
    def cutData(self,left=None,right=None,innercut=False):
        data_changed=False
        if self.data_before_first_selection is not None:
            previous_data=self.data
            previous_wave=self.wavelength
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
            data_changed=True
        if self.data_before_cut is None:
            self.preprocess_before_cut=self.general_report['Preprocessing']
            self.data_before_cut=self.data
            self.wavelength_before_cut=self.wavelength
            self.x_before_cut=self.x
        if self.wavelength is None:
            wavelength=np.array([i for i in range(len(self.data[1]))])
        else:
            wavelength=self.wavelength
        statement=f'\t\tplease select only left or right, if booth an innercut will be done if innercut is set to True'
        if innercut==False and left is None and right is None:
            print(statement)
            return statement
        if innercut==False:
            if left is not None:
                assert right is None,statement
                cut_index=(pd.Series(wavelength)-left).abs().sort_values().index[0]
                if self.wavelength is not None:
                    self.wavelength=self.wavelength[:cut_index]
                else:
                    self.wavelength=wavelength[:cut_index]
                self.data=self.data[:,:cut_index]
                action=f'\t\tSelected data from {left} {self.wavelength_unit}'
            if right is not None:
                assert left is None,statement
                cut_index=(pd.Series(wavelength)-right).abs().sort_values().index[0]
                if self.wavelength is not None:
                    self.wavelength=self.wavelength[cut_index:]
                else:
                     self.wavelength=wavelength[cut_index:]
                self.data=self.data[:,cut_index:]
                action=f'\t\tSelected data until {right} {self.wavelength_unit}'
        elif innercut=='select':
            assert left is not None and right is not None, 'to select an area left and right margins should be given'
            cut_right=(pd.Series(wavelength)-right).abs().sort_values().index[0]
            cut_left=(pd.Series(wavelength)-left).abs().sort_values().index[0]
            if self.wavelength is not None:
                   self.wavelength= self.wavelength[cut_left:cut_right]
            else:
                    self.wavelength= self.wavelength[cut_left:cut_right]
            self.data=self.data[:,cut_left:cut_right]
            self.curve_resultados=self.data*0.0
            action=f'\t\tSelected data from {left} {self.wavelength_unit} until {right} {self.wavelength_unit}'
        else:
            assert left is not None and right is not None, 'to do an inner cut left and right margins should be given'
            cut_right=(pd.Series(wavelength)-right).abs().sort_values().index[0]
            cut_left=(pd.Series(wavelength)-left).abs().sort_values().index[0]
#            if self.wavelength is not None:
#                   self.wavelength[cut_left:cut_right]=self.wavelength[cut_left:cut_right]*0.0
##                   self.wavelength=np.append(self.wavelength[:cut_left],self.wavelength[cut_right:])
#            else:
#                    self.wavelength[cut_left:cut_right]=self.wavelength[cut_left:cut_right]*0.0
#                    self.wavelength=np.append(wavelength[:cut_left], wavelength[cut_right:])
            self.data[:,cut_left:cut_right]=self.data[:,cut_left:cut_right]*0.0
#            self.data=np.concatenate((self.data[:,:cut_left],self.data[:,cut_right:]),axis=1)
            self.curve_resultados=self.data*0.0
            self.inner_cut_done=[left,right]
            action=f'\t\tCutted data from {left} {self.wavelength_unit} until {right} {self.wavelength_unit}'
        self.general_report['Preprocessing']['Cutted Wavelengths'].append(action)
        self.general_report['Sequence of actions'].append(f'\t--> Cut or selection of wavelength')
        self.general_report['Data Shape']['Actual number of traces']=self.data.shape[1]
        if data_changed:
            self.data_before_first_selection=self.data
            self.wavelength_before_first_selection=self.wavelength
            self.data=previous_data
            self.wavelength=previous_wave
    
    def cutTimeData(self,mini=None,maxi=None):
        data_changed=False
        if self.data_before_first_selection is not None:
            previous_data=self.data
            self.data=self.data_before_first_selection
            data_changed=True
        if self.data_before_time_cut is None:
            self.preprocess_before_time_cut=self.general_report['Preprocessing']
            self.data_before_time_cut=self.data
            self.x_before_time_cut=self.x
            self.wavelength_before_time_cut=self.wavelength
        if mini is not None and maxi is None:
            min_index=np.argmin([abs(i-mini) for i in self.x])
            maxi_index=None
            action=f'\t\tSelected data from {mini} {self.time_unit}'
        elif maxi is not None and mini is None:
            maxi_index=np.argmin([abs(i-maxi) for i in self.x])
            min_index=0
            action=f'\t\tSelected data until {maxi} {self.time_unit}'
        else:
            min_index=np.argmin([abs(i-mini) for i in self.x])
            maxi_index=np.argmin([abs(i-maxi) for i in self.x])
            action=f'\t\tSelected data from {mini} {self.time_unit} until {maxi} {self.time_unit}'
        if maxi_index is not None:
            self.x=self.x[min_index:maxi_index+1]
            self.data=self.data[min_index:maxi_index+1,:]
        else:
            self.x=self.x[min_index:]
            self.data=self.data[min_index:,:]
        self.general_report['Preprocessing']['Cutted Times'].append(action)
        self.general_report['Sequence of actions'].append(f'\t--> Cut or selection of time range')
        self.general_report['Data Shape']['Actual time points']=self.data.shape[0]
        if data_changed:
            self.data_before_first_selection=self.data
            if maxi_index is not None:
                self.data=previous_data[min_index:maxi_index+1,:]
            else:
                self.data=previous_data[min_index:,:]
                
    def averageTimePoints(self,starting_point, step, method='log',grid_dense=5):
        data_changed=False
        if self.data_before_first_selection is not None:
            previous_data=self.data
            previous_wave=self.wavelength
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
            data_changed=True
        if self.data_before_average_time is None:
            self.preprocess_average_time=self.general_report['Preprocessing'] 
            self.data_before_average_time=self.data
            self.x_before_average_time=self.x
            self.wavelength_before_average_time=self.wavelength
        point=np.argmin([abs(i-starting_point) for i in self.x])
        time_points=[i for i in self.x]
        value=step
        point=np.argmin([abs(i-starting_point) for i in self.x])
        number=time_points[point]+step
        index=[]
        it=point+1
        if method == 'log':
            log=1
            value/=grid_dense
        while number<time_points[-1]:
            time_average=[i for i in range(it+1,len(time_points)) if time_points[i]<number] #selecting points to average the range is from 179 to end as in the first 180 point there is nothing to average # the value 100 gives the diference in time which are average   
            if method == 'log':
                log+=1
            number +=value*log 
            if len(time_average)>=1:
                index.append(time_average)
                it=time_average[-1]
            else:
                log+=1
        index.append([i for i in range(index[-1][-1],len(time_points))])    
         
        data=self.data[:point+1,:]
        time=self.x[:point+1]
        if method == 'log':
            method=f'a growing step initially of'
        else:
            method=f'constant'
        self.general_report['Preprocessing']['Average time points'].append(f'\t\tAverage from {starting_point}, with {method} {step} step')
        self.general_report['Sequence of actions'].append(f'\t--> Average of time points')
        self.general_report['Data Shape']['Actual time points']=self.data.shape[0]
        if data_changed:
            previous=previous_data[:point+1,:]
        for i in range(len(index)):
            column=np.mean(self.data[index[i][0]:index[i][-1]+1,:],axis=0).reshape(1,data.shape[1])
            timin=np.mean(self.x[index[i][0]:index[i][-1]+1])
            data=np.concatenate((data,column),axis=0)
            time=np.append(time,timin)
            if data_changed:
                coldata=np.mean(previous_data[index[i][0]:index[i][-1]+1,:],axis=0).reshape(1,previous.shape[1])
                previous=np.concatenate((previous,coldata),axis=0)
        self.data=data
        self.x=time
        if data_changed:
            self.data_before_first_selection=self.data
            self.wavelength_before_first_selection=self.wavelength
            self.data=previous
            self.wavelength=previous_wave
            
    def selectTracesGrapth(self):
        if self.data_before_first_selection is None:
            self.data_before_first_selection=self.data
            self.wavelength_before_first_selection=self.wavelength
        else:
            self.data=self.data_before_first_selection
            self.curve_resultados=self.data*0.0
            self.wavelength=self.wavelength_before_first_selection
        fig,a=self.plotSpectra(select=True)
        self.average_trace=0
        self.data=np.take(self.data, self.selected_traces, axis=1)
        self.curve_resultados=np.take(self.curve_resultados, self.selected_traces, axis=1)
        plt.close(fig)
        self.SVD_fit=False
        if self.wavelength is not None:
            self.wavelength=np.take(self.wavelength, self.selected_traces)
        if  self.params_initialized==True:
            self.paramsAlready()    
    
    def getAutoPoints(self,spectra=8,wave=None,rango=None, include_rango_max=True):
        if self.data_before_first_selection is not None:
            data=self.data_before_first_selection
            wavelength=self.wavelength_before_first_selection
        else:
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
            if self.deconv and rango is None:
                point=[idx[0]+np.argmin(abs(data[idx[0]:,idx[1]]-i)) for i in np.linspace(np.min(data[idx[0]:,idx[1]]),np.max(data[idx[0]:,idx[1]]),spectra)]
            else:
                point=[0+np.argmin(abs(data[:,idx[1]]-i)) for i in np.linspace(np.min(data[:,idx[1]]),np.max(data[:,idx[1]]),spectra)]
                if 0 not in point:
                        point[np.argmin(point)]=0
            if rango is not None and include_rango_max:
                     point[np.argmax(point)]=-1
            print (wavelength[idx[1]])
            return np.sort(np.array(x)[point])
        else:
            if wavelength is not None:
                wave_idx=np.argmin(abs(np.array(wavelength)-wave))
                idx=np.argmax(abs(data[:,wave_idx]))
                if self.deconv and rango is None:    
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
        
    def reDoFitNTimes(self,n_fit, *taus,vary_taus='original', maxfev=None, r_seed=True, apply_weights=False):
        col_name=[x for i in range(1,4) for x in ('tau_%i initial' % i,'tau_%i final' % i)]\
                                              +['N° Iterations','red Xi^2','success']
        if r_seed:
            np.random.seed(42) 
        if vary_taus=='original':
            vary_taus=self.initial_vary_taus                                 
        self.NFit_result=pd.DataFrame(columns=col_name)
        n_arg=len(list(taus))
        assert n_arg == self.exp_no, f'number of exponential ranges is {self.exp_no} you defined a {n_arg} expfunction'
        for fit in range(n_fit):        
            initial_times=[np.random.randint(taus[i][0],taus[i][1]) for i in range (self.exp_no)]
            self.initialParams2(self.initial_t0,initial_times,fwhm=self.initial_fwhm,vary_t0=self.initial_vary_t0,opt_fwhm=self.initial_opt_fwhm,GVD_corrected=self.initial_GVD_corrected)
            self.preFit()
            self.finalFit(vary_taus,maxfev=maxfev,time_constraint=self.initial_time_constraint,save_fit=False,apply_weights=apply_weights)
            final_time=[self.params['tau%i_1' %(i+1)].value for i in range (self.exp_no)]
            self.NFit_result.loc[fit]=[x[i] for x in [(i, ii) for i, ii in zip(initial_times,final_time) ]
                    for i in range(len(x))]+[float(self.resultados.nfev),float(self.resultados.redchi),self.resultados.success]
        return self.NFit_result
    
    def dataBootStrap(self,n_boot):
        col_name=[x for i in range(1,self.exp_no+1) for x in ('tau_%i initial' % i,'tau_%i final' % i)]\
                                              +['N° Iterations','red Xi^2','success']
        self.data_boostrap_result=pd.DataFrame(columns=col_name)                                    
        initial_time=self.initial_taus
        final_time=[self.params['tau%i_1' %(i+1)].value for i in range (self.exp_no)]
        self.data_boostrap_result.loc[0]=[x[i] for x in [(i, ii) for i, ii in zip(initial_time,final_time) ]
                    for i in range(len(x))]+[float(self.resultados.nfev),float(self.resultados.redchi),self.resultados.success]
        
        self.data_set_boot=self.data_before_last_Fit.copy()
        for i in range(n_boot):
            new_data=self.data*0.0
            index=np.random.choice(np.linspace(0,len(self.data[1])-1,len(self.data[1])),len(self.data[1]))
            for i,ii in enumerate(index):
                new_data[:,i]=  self.data[:,int(ii)]
            self.data_set_boot=np.dstack((self.data_set_boot,new_data))
        self.data_set_boot=self.data_set_boot[:,:,1:]
        for boot in range(n_boot):     
            self.data = self.data_set_boot[:,:,boot]
#            fwhm=0.16,vary_t0=False,opt_fwhm=False,GVD_corrected=True
            self.initialParams2(self.initial_t0,self.initial_taus,fwhm=self.initial_fwhm,vary_t0=self.initial_vary_t0,opt_fwhm=self.initial_opt_fwhm,GVD_corrected=self.initial_GVD_corrected)
            self.preFit()
            self.finalFit(self.initial_vary_taus,maxfev=self.initail_maxfev,time_constraint=self.initial_time_constraint,save_fit=False)
            final_time=[self.params['tau%i_1' %(i+1)].value for i in range (self.exp_no)]
            self.data_boostrap_result.loc[boot+1]=[x[i] for x in [(i, ii) for i, ii in zip(initial_time,final_time) ]
                    for i in range(len(x))]+[float(self.resultados.nfev),float(self.resultados.redchi),self.resultados.success]
            print(f'the number of boots is: {boot}')
        return self.data_boostrap_result 
    
    def residuesBootStrap(self,fit_number,n_boots,size,save=True, cal_conf=True, new=False):
        assert size==10 or size==15 or size==20 or size==25 or size==33, 'Size should be either 10, 15, 20, 25, 33, this is the values in percentage that will be randomly changed' 
        if fit_number in self.bootstrap_residues_record.keys() and new == False:
            size=self.bootstrap_residues_record[0]
            print(f'size is fixed to {size} as there are alredy bootstrap run with this value')
        if size==10:
            div=10
        elif size==15:
            div=7
        elif size==20:
            div=5
        elif size==25:
            div=4
        elif size==33:
            div=3    
        fit=self.all_fit[fit_number]
        self.x_before_last_Fit=fit[0]
        self.data_before_last_Fit=fit[1]
        self.wavelength_before_last_Fit=fit[2]
        self.resultados=fit[3]
        self.exp_no=fit[4]
        self.deconv=fit[5]
        self.tau_inf=fit[6]
        self.type_fit=fit[8][2]
        maxfev=fit[8][0]
        time_constraint=fit[8][1]
        if type(fit[8][3]) == str:
            self.weights['apply']=False
            apply_weight=False
        elif type(fit[8][3]) == dict:
            self.weights=fit[8][3]
            apply_weight=True
        self.params=deepcopy(self.resultados.params)
        if self.type_fit == 'Exponential':
            if self.deconv:
                names=['t0_1','fwhm_1']+['tau%i_1'%(i+1) for i in range(self.exp_no)]
                extra=2
            else:
                names=['t0_1']+['tau%i_1'%(i+1) for i in range(self.exp_no)]
                extra=1
            initial_params=[self.params[name].init_value for name in names]
            initial_variations=[self.params[name].vary for name in names]                                    
            col_name=[x for ii,i in enumerate(names) for x in (i.split('_')[0]+' initial',i.split('_')[0]+' final') if initial_variations[ii]]\
                                +['N° Iterations','red Xi^2','success']
            initial_values=[initial_params[i] for i in range(len(initial_params)) if initial_variations[i]]  
        else:
            if self.deconv:
                names=['t0_1','fwhm_1']+['k_%i%i' % (i+1,i+1) for i in range(self.exp_no)]
                extra=2
            else:
                names=['t0_1']+['k_%i%i' % (i+1,i+1)  for i in range(self.exp_no)]
                extra=1    
            col_names=[i.split('_')[0] if ii < extra else i for ii,i in enumerate(names)]
            initial_params=[self.params[name].init_value for name in names]
            initial_variations=[self.params[name].vary for name in names]                                    
            col_name=[x for ii,i in enumerate(col_names) for x in (i+' initial',i+' final') if initial_variations[ii]]\
                                +['N° Iterations','red Xi^2','success']
            initial_values=[initial_params[i] for i in range(len(initial_params)) if initial_variations[i]]  
        self.residues_boostrap_result=pd.DataFrame(columns=col_name) #create result data frame fro appending bootstrap fit values
        final_time=[self.params[name].value for name in names if self.params[name].vary]
        if self.type_fit != 'Exponential':
            final_time=[abs(ii) if i >= 1 else ii for i,ii in enumerate(final_time)]
            initial_values=[abs(ii) if i >= 1 else ii for i,ii in enumerate(initial_values)]
#        final_time=[self.params['tau%i_1' %(i+1)].value for i in range (self.exp_no)]
        self.residues_boostrap_result.loc[0]=[x[i] for x in [(i, ii) for i, ii in zip(initial_values,final_time) ]
                    for i in range(len(x))]+[float(self.resultados.nfev),float(self.resultados.redchi),self.resultados.success]
        fittes=self.results(self.params)
        self.residue_set_boot=self.data_before_last_Fit.copy()
        for boot in range(n_boots):
            residues = 0.0*self.data_before_last_Fit[:]
            for ii in range(len(residues[1])):
                residues[:, ii] = self.data_before_last_Fit[:, ii] - fittes[:,ii]
            for it in range(len(residues[1])//div):
                value1=np.random.randint(len(residues[1]))
                value2=np.random.randint(len(residues[1]))
                residues[:,value1]=residues[:,value2]
            data2=0.0*self.data_before_last_Fit[:]
            for da in range(len(residues[1])):
                data2[:, da]=fittes[:,da]+residues[:,da]
            self.residue_set_boot=np.dstack((self.residue_set_boot,data2))
        self.residue_set_boot=self.residue_set_boot[:,:,1:]
        initial_prams=deepcopy(self.params)
        for i in initial_prams:
            initial_prams[i].value=initial_prams[i].init_value
        self.prefit_done=True
        for boot in range(n_boots):     
            self.data = self.residue_set_boot[:,:,boot]
            self.params=deepcopy(initial_prams)
            self.finalFit(initial_variations[extra:],maxfev=maxfev,time_constraint=time_constraint,save_fit=False,apply_weights=apply_weight)
            final_time=[self.params[name].value for name in names if self.params[name].vary]
            if self.type_fit != 'Exponential':
                final_time=[abs(ii) if i >= 1 else ii for i,ii in enumerate(final_time)]
            self.residues_boostrap_result.loc[boot+1]=[x[i] for x in [(i, ii) for i, ii in zip(initial_values,final_time) ]
                    for i in range(len(x))]+[float(self.resultados.nfev),float(self.resultados.redchi),self.resultados.success]
            print(f'the number of boots is: {boot}')
        if save:
            if fit_number in self.bootstrap_residues_record.keys() and new == False:
                first=self.bootstrap_residues_record[fit_number][1]
                all_boot=pd.concat([first,self.residues_boostrap_result.iloc[1:,:]])
                self.bootstrap_residues_record[fit_number][1]=all_boot
            else:
                self.bootstrap_residues_record[fit_number]=[size,self.residues_boostrap_result.copy()]
            if cal_conf:
                self.bootConfInterval(fit_number=fit_number)
        return self.residues_boostrap_result     
    
    def bootConfInterval(self,fit_number=None,data=None):
        if fit_number is not None:
            if fit_number in self.bootstrap_residues_record.keys():
                data=self.bootstrap_residues_record[fit_number][1]
            else:
                return 'No bootstrap run for this fit'
        assert data is not None
        names=[i for i in data.keys() if 'final' in i]
        values=[0.27,4.55,31.7,-1,68.27,95.45,99.73]
        table=pd.DataFrame(columns=['99.73%','95.45%','68.27%','_BEST_','68.27%','95.45%','99.73%'])
        print('conf calc')
        for i in names:
            array=data[i].values
            line=[np.percentile(array,val)-array[0] if val !=-1 else array[0] for val in values]
            table.loc[i.split(' ')[0]]=line
        if fit_number is not None:
            self.bootstrap_residues_record[fit_number].append(table)
        else:
            return table

    def plotBootStrapResults(self,fit_number,param,kde=False):
        fig, axes = plt.subplots(1, 1)
        bootsTrap=self.bootstrap_residues_record[fit_number][1]
        names=[i.split(' ')[0] for i in bootsTrap.keys() if 'final' in i]
        stats=bootsTrap.describe()
        stats_values={}
        for name in names:    
            stats_values[name+' mean']=round(stats[name+' final']['mean'],4)
            stats_values[name+' std']=round(stats[name+' final']['std'],4)
        axes = distplot(bootsTrap[param+' final'].values,rug=False,norm_hist =False,kde=kde,hist_kws=dict(edgecolor="k", linewidth=2))
        plt.xlabel(f'Time ({self.time_unit})')
        maxi=bootsTrap[param+' final'].max()
        mini=bootsTrap[param+' final'].min()
        mean=bootsTrap[param+' final'].mean()#[0]
        dif_max=abs(maxi-mean)
        dif_min=abs(mini-mean)
        if kde==False:
            plt.ylabel('Counts')
            plt.xlim(mini-maxi*0.1,maxi+maxi*0.1)
        else:
            plt.ylabel('Density function')
        if dif_max>dif_min:
            pos=1
        else:
            pos=2
        mean=stats_values[param+' mean']
        std=stats_values[param+' std']
        texto= AnchoredText (s=f'$\mu={mean}$ {self.time_unit}\n $\sigma={std}$ {self.time_unit}',loc=pos)
        axes.add_artist(texto)
        return fig, axes
    
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
             self.dispersion_BK7=self.dispersion(self.wavelength,self.BK7_param,self.excitation)
        if self.dispersion_Caf2 is 0:
            self.dispersion_Caf2=self.dispersion(self.wavelength,self.CaF2_param,self.excitation)
        if self.dispersion_SiO2 is 0:
            self.dispersion_SiO2=self.dispersion(self.wavelength,self.SiO2_param,self.excitation)
        print(CaF2,SiO2,BK7,offset)
        self.gvd=self.dispersion_BK7*BK7+self.dispersion_SiO2*SiO2+self.dispersion_Caf2*CaF2+offset
        self.CaF2,self.BK7,self.SiO2,self.GVD_offset=CaF2,BK7,SiO2,offset
        return self.gvd
    
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]
    
    def correctGVD(self,verified=False):
        if self.data_before_first_selection is not None:
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
        if self.data_before_GVD is None:
            self.preprocess_before_GVD=self.general_report['Preprocessing'] 
            self.data_before_GVD=self.data
            self.x_before_GVD=self.x
            self.wavelength_before_GVD=self.wavelength
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
                            if sub == inf: sub=self.x[idex+2] #in case to consecutive points have identical time value
                            w=abs((sub-valor)/(sub-inf))
                            corrected_data[ii,i]=w*self.data[idex,i]+(1-w)*self.data[idex+1,i]    
                    else:
                        valores[ii,i]=+1
                        inf = self.x[idex-1]
                        sub = self.x[idex]
                        if sub ==  inf: inf=self.x[idex-2] #in case to consecutive points have identical time value
                        w=abs((valor-sub)/(sub-inf))
                        corrected_data[ii,i]=w*self.data[idex-1,i]+(1-w)*self.data[idex,i]
        self.GVD_correction='in process'
#        fig, ax = plt.subplots(figsize=(6,6))
#        pcolormesh(self.wavelength,self.x[:46],pd.DataFrame(corrected_data).iloc[:46].values,cmap='RdYlBu')
        self.corrected_data=corrected_data
        if verified:
            self.verifiedGVD()
        else:
            if self.polynomGVD:
               polynom=[f'{round(i,2)}X^{ii}' if ii>1 else f'{i}' for ii,i in enumerate(self.polynom_gvd[::-1]) if i is not 0]
               polynom=('+'.join(polynom[::-1])).replace('+-','-')
               self.general_report['Preprocessing']['GVD correction'].append(f'\t\tCorrected with Polynom {polynom}')
               self.general_report['Sequence of actions'].append(f'\t--> Correction of GVD with polynom')
            else:
               self.general_report['Preprocessing']['GVD correction'].append(f'\t\tCorrected with Sellmeier equation: {round(self.GVD_offset,2)} offset,\n\t\t  SiO2:{round(self.SiO2,2)} mm, CaF2:{round(self.CaF2,2)} mm BK7:{round(self.BK7,2)} mm')
               self.general_report['Sequence of actions'].append(f'\t--> Correction of GVD with Sellmeier equation')
            self.data=self.corrected_data
    
#    def GVDFromPolynom(self,qt=None):
#        self.selectTraces('auto',0,False)
#        self.number_fig=0
#        self.point_pol_GVD=[]
#        self.GVDpolynomFigs()
#        if qt is not None:
#            self.qt_path=qt
#     
#    def GVDpolynomFigs(self):
#        self.fig,a0 = plt.subplots(1,figsize=(7,6))
#        xlabel='Time ('+self.time_unit+')'
#        alpha,s=0.60,8
#        value=2
#        if self.time_unit=='ns':
#            value=value/1000
#        self.index2ps=np.argmin([abs(i-value) for i in self.x])
#        legenda=f'{self.wavelength[self.number_fig]} {self.wavelength_unit}'
#        data=self.data[:self.index2ps,self.number_fig]
#        a0.plot(self.x[:self.index2ps],data,marker='o',alpha=alpha,ms=4,ls='')
#        a0.set_ylim(np.min(data)-abs(np.min(data)*0.1),np.max(data)+np.max(data)*0.1)
#        a0.set_xlim(self.x[0]-self.x[self.index2ps]/50,self.x[self.index2ps]+self.x[self.index2ps]/50)
#        a0.axhline(linewidth=1,linestyle='--', color='k')
#        a0.legend([legenda])
#        a0.ticklabel_format(style='sci',axis='y')
#        #f.tight_layout()
#        a0.set_ylabel('$\Delta$A',size=14)
#        a0.set_xlabel(xlabel,size=14)
#        a0.minorticks_on()
#        a0.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
#        plt.subplots_adjust(bottom=0.2)
#        self.cursor_pol=SnaptoCursor( a0,self.x[:self.index2ps], data,number_click=1)
#        self.fig.canvas.mpl_connect('button_press_event', self.cursor_pol.onClick)
#        self.fig.canvas.mpl_connect('motion_notify_event', self.cursor_pol.mouseMove)
#        self.fig.canvas.mpl_connect('axes_enter_event', self.cursor_pol.onEnterAxes)
#        self.fig.canvas.mpl_connect('axes_leave_event', self.cursor_pol.onLeaveAxes)
#        resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
#        if self.number_fig==8:
#            self.button = Button(resetax, 'Finish', color='tab:red', hovercolor='0.975')
#            self.button.on_clicked(self.fitPolGVD)
#        else:   
#            self.button = Button(resetax, 'Next', color='tab:red', hovercolor='0.975')
#            self.button.on_clicked(self.nextFigPol)
#        if self.qt_path is not None:
#            thismanager = plt.get_current_fig_manager()
#            thismanager.window.setWindowIcon(QIcon(self.qt_path))
#        self.fig.show()
#        
#    def nextFigPol(self,event):
#        if len(self.cursor_pol.datax)==0:
#            pass
#        else:
#            self.number_fig += 2
#            self.point_pol_GVD.append(self.cursor_pol.datax[0])
#            plt.close(self.fig)
#            self.GVDpolynomFigs()
            
    def GVDFromPolynom(self,qt=None):
        if self.data_before_first_selection is not None:
            self.data=self.data_before_first_selection
            self.wavelength=self.wavelength_before_first_selection
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
           self.data=self.corrected_data
           self.GVD_correction=True
           print('Data has been corrected from GVD')
           plt.close(self.fig)
           plt.close()
           plt.close(self.figGVD)
           if self.polynomGVD:
               polynom=[f'{round(i,2)}X^{ii}' if ii>1 else f'{i}' for ii,i in enumerate(self.polynom_gvd[::-1]) if i is not 0]
               polynom=('+'.join(polynom[::-1])).replace('+-','-')
               self.general_report['Preprocessing']['GVD correction'].append(f'\t\tCorrected with Polynom {polynom}')
               self.general_report['Sequence of actions'].append(f'\t--> Correction of GVD with polynom')
           else:
               self.general_report['Sequence of actions'].append(f'\t--> Correction of GVD with Sellmeier equation')
               self.general_report['Preprocessing']['GVD correction'].append(f'\t\tCorrected with Sellmeier equation: {round(self.GVD_offset,2)} offset,\n\t\t  SiO2:{round(self.SiO2,2)} mm, CaF2:{round(self.CaF2,2)} mm BK7:{round(self.BK7,2)} mm')
        else:
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
    
    def verifiedFit(self,fit_number=None):
        if fit_number is not None:
            self.x_verivefit=self.all_fit[fit_number][0]
            self.data_fit=self.all_fit[fit_number][1]
            self.wavelength_fit=self.all_fit[fit_number][2]
            if self.all_fit[fit_number][9]:
                params=self.all_fit[fit_number][11]
            else:
                params=self.all_fit[fit_number][3].params
            self.deconv=self.all_fit[fit_number][5]
        else:
            self.data_fit=self.data_before_last_Fit
            self.x_fit=self.x_before_last_Fit
            self.wavelength_fit=self.wavelength_before_last_Fit
            params=self.params
            self.deconv=self.deconv
        xlabel='Time ('+self.time_unit+')'
        self.fig, ax = plt.subplots(2, 1,sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [1, 5]})
        self.fittes=self.results(params)
        self.residues= 0.0*self.data_fit[:]
        for i in range(self.fittes.shape[1]):
            self.residues[:,i]=self.data_fit[:,i]-self.fittes[:,i]
        initial_i=0
        self.l,=ax[1].plot(self.x_verivefit,self.data_fit[:,initial_i],marker='o',ms=3,linestyle=None,label='raw data')
        self.lll,=ax[0].plot(self.x_verivefit,self.residues[:,initial_i],marker='o',ms=3,linestyle=None,label='residues')
        self.ll,=ax[1].plot(self.x_verivefit,self.fittes[:,initial_i],alpha=0.5,lw=1.5,color='r',label='fit')
        delta_f=1.0
        _,maxi=self.data_fit.shape
        axcolor='orange'
        axspec = self.fig.add_axes([0.20, .02, 0.60, 0.01],facecolor=axcolor)
        self.sspec = Slider(axspec, 'curve number', 0, maxi-1,valstep=delta_f,valinit=0)
        self.sspec.on_changed(self.updateVerifiedFit)
        ax[0].ticklabel_format(style='sci',axis='y')
        ax[1].ticklabel_format(style='sci',axis='y')
        ax[1].minorticks_on()
        ax[1].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
        ax[0].minorticks_on()
        ax[0].axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=14)
        ax[0].set_ylim(np.min(self.residues)-abs(np.min(self.residues)*0.1),np.max(self.residues)+np.max(self.residues)*0.1)
        ax[1].set_ylim(np.min(self.data_fit)-abs(np.min(self.data_fit)*0.1),np.max(self.data_fit)+np.max(self.data_fit)*0.1)
        #f.tight_layout()
        ax[1].legend(loc='upper right')
        ax[0].legend(loc='upper right')
        title=round(self.wavelength_fit[initial_i])
        plt.title(f'{title} nm')
        ax[1].set_xlabel(xlabel, size=14)
        plt.xlim(self.x_verivefit[0]-self.x_verivefit[-1]/50,self.x_verivefit[-1]+self.x_verivefit[-1]/50)
        ax[1].set_ylabel(r'$\Delta$A',size=14)
        ax[0].set_ylabel('Residues',size=14)
        self.fig.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        return self.fig,ax
        
    def updateVerifiedFit(self,val):
        # amp is the current value of the slider
        value = self.sspec.val
        value=int(round(value))
        # update curve
        title=round(self.wavelength_fit[value])
        plt.title(f'{title} nm')
        self.l.set_ydata(self.data_fit[:,value])
        self.ll.set_ydata(self.fittes[:,value])
        self.lll.set_ydata(self.residues[:,value])
        # redraw canvas while idle
        self.fig.canvas.draw_idle()

    def plotConcentrations(self,fit_number=None,size=14,names=None,plot_total_C=True,legend=True):#tmp function.
        if fit_number is not None:
            #verify type of fit is: either fit to Singular vectors or global fit to traces
            if self.all_fit[fit_number][9]:
                params=self.all_fit[fit_number][11]
                data=self.all_fit[fit_number][10]
            else:
                params=self.all_fit[fit_number][3].params
                data=self.all_fit[fit_number][1]
            exp_no=self.all_fit[fit_number][4]
            type_fit=self.all_fit[fit_number][8][2]#check for type of fit done
            x=self.all_fit[fit_number][0]
        else:
            data=self.data_before_last_Fit
            exp_no=self.exp_no
            type_fit=self.type_fit
            params=self.params
            x=self.x
        if type_fit =='Expeonential':
            return 'This fucntion is only available for target fit'
        else:
            xlabel='Time ('+self.time_unit+')'
            ndata, nx = data.shape
    #        resid = 0.0*self.data[:]
            #take kmatrix shit from params and solve eqs
            maxi_tau=-1/params['k_%i%i' % (exp_no-1,exp_no-1)].value
            if maxi_tau > x[-1]:
                maxi_tau = x[-1]
            ksize = exp_no #size of the matrix = no of exponenses = no of species
            kmatrix = np.array([[params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
            cinitials = [params['c_%i' % (i+1)].value for i in range(ksize)]
            eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
            #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
            eigenmatrix = np.array(vects) 
            coeffs = np.linalg.solve(eigenmatrix, cinitials) #solve the initial conditions sheet            
            t0 = params['t0_1'].value
            fwhm = params['fwhm_1'].value
            expvects = [coeffs[i]*self.expGaussConv(x-t0,-eigs[i],fwhm/2.35482) for i in range(len(eigs))] 
            concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
            if names is None or len(names) != exp_no:
                names=[f'Specie {i}' for i in range(exp_no)]
            fig,ax = plt.subplots(1,figsize=(8,6))
            for i in range(len(eigs)):
                ax.plot(x, concentrations[i],label=names[i])
            allc = sum(concentrations)
            if plot_total_C:
                allc = sum(concentrations)
                ax.plot(x, allc, label='Total concetration') #sum of all for checking => should be unity
            if legend:
                plt.legend(loc='best')
            ax.minorticks_on()
            ax.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
            plt.xlabel(xlabel,size=size)
            plt.ylabel('Concentration (A.U.)',size=size)
            plt.xlim(-3,round(maxi_tau*7))
            return fig,ax
    
    def initialParamsModel(self,model_params,t0,vary_t0=False,fwhm=0.16, vary_fwhm=False):
        self.type_fit='Target'
        self.last_model_params=model_params.copy()
        self.initial_params = model_params.copy()
        self.exp_no = self.initial_params["exp_no"].value
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        ndata, nx = self.data.shape
        self.tau_inf = None
        for iy in range(nx):
            self.initial_params.add_many(('y0_' +str(iy+1), 0,True, None, None, None, None), #i fixed, may unfix later
                            ('t0_'+str(iy+1), t0, vary_t0,  np.min(self.x), None, None, None))
            if self.deconv:
                self.initial_params.add('fwhm_' +str(iy+1), fwhm, vary_fwhm, 0.000001, None, None, None)
            
            for i in range (self.exp_no):
                # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                self.initial_params.add('pre_exp%i_' %(i+1) +str (iy+1), 0.1*10**(-i), True, None, None, None, None)        
        
#         self.t0_vary=vary_t0
#        self.params['t0_1'].value=t0
#        self.params['fwhm_1'].value=fwhm
        for i in range(self.exp_no):
            if(self.initial_params['k_%i%i' % (i+1,i+1)].value != 0):
                self.initial_params.add('tau%i_1' % (i+1),-1/self.initial_params['k_%i%i' % (i+1,i+1)].value,vary=False)
            else:
                self.resultados.params.add('tau%i_1' % (i+1),np.inf,vary=False)
        for iy in range(2,self.data.shape[1]+1):
            if self.deconv:
                self.initial_params['fwhm_%i'% iy].expr='fwhm_1'   
                if self.GVD_correction != True:
                    self.initial_params['t0_%i' % iy].expr=None
                    self.initial_params['t0_%i' % iy].vary=True
                    self.initial_params['t0_%i' % iy].value=t0
                    self.t0_vary=True
                else:
                    self.t0_vary=vary_t0
                    self.initial_params['t0_%i' % iy].expr='t0_1'
            else:
                 self.initial_params['t0_%i' % iy].vary=False
                 self.initial_params['t0_%i' % iy].expr='t0_1' 
                 self.t0_vary=False
        self.general_report['Sequence of actions'].append('\t--> New Target Model parameters initialized')         
        self.params_initialized=True
        self.Fit_completed=False
        self.prefit_done=False
#    def objectiveTF(self,params):
#        #changed by lucas lines 189
#        """ calculate total residual for fits to several data sets held
#        in a 2-D array"""
#        ndata, nx = self.data_before_last_Fit.shape
#        resid = 0.0*self.data_before_last_Fit[:]
#
#        #take kmatrix shit from params and solve eqs
#        ksize = self.exp_no #size of the matrix = no of exponenses = no of species
#        kmatrix = np.array([[params['k_%i%i' % (i+1,j+1)].value for j in range(ksize)] for i in range(ksize)])
#        cinitials = [params['c_%i' % (i+1)].value for i in range(ksize)]
#        eigs, vects = np.linalg.eig(kmatrix)#do the eigenshit
#        #eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))]) 
#        eigenmatrix = np.array(vects) 
#        coeffs = np.linalg.solve(eigenmatrix, cinitials) #solve the initial conditions sheet
#        
#        if self.deconv:
#            if(self.GVD_correction):
#            
#                t0 = params['t0_1'].value
#                fwhm = params['fwhm_1'].value
#                expvects = [coeffs[i]*self.expGaussConvFast(self.x-t0,-eigs[i],fwhm/2.35482) for i in range(len(eigs))] 
#                concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
#                
#                #plt.figure()
#                #for i in range(len(eigs)):
#                #    plt.plot(self.x, concentrations[i])
#                #plt.show()
#                
#                for i in range(nx):
#                    resid[:, i] = self.data_before_last_Fit[:, i] - self.expNGaussDatasetFast(params, i, concentrations)  
#         
#            else:  #didnt tested but should work, if no then probably minor correction is needed.
#                for i in range(nx):                
#                    t0 = params['t0_%i' % i+1].value
#                    fwhm = params['fwhm_%i' % i+1].value
#                    expvects = [coeffs[i]*self.expGaussConvFast(self.x-t0,-eigs[i],fwhm/2.35482) for i in range(len(eigs))] 
#                    concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
#
#                    resid[:, i] = self.data_before_last_Fit[:, i] - self.expNGaussDatasetFast(params, i, concentrations)                  
#        
#       
#        else:                             
#            t0 = params['t0_%i' % i+1].value
#            index=np.argmin([abs(i-t0) for i in self.x])
#            resid = 0.0*self.data_before_last_Fit[index:,:]
##            fwhm = 0.0000001 #lets just take very short IRF, effect should be the same
#            expvects = [coeffs[i]*self.exp1(self.x[index:]-t0,-eigs[i]) for i in range(len(eigs))] 
#            concentrations = [sum([eigenmatrix[i,j]*expvects[j] for j in range(len(eigs))]) for i in range(len(eigs))]
#            for i in range(nx):  
#                resid[:, i] = self.data_before_last_Fit[:, i] - self.expNDatasetFast(params, i, concentrations)          
#                       
#        # now flatten this to a 1D array, as minimize() needs
#        self.number_it=self.number_it+1
#        if(self.number_it % 100 == 0):
#            
#            #print(vects)
#            #print(eigenmatrix)
#            #print(params['pre_exp%i_' %(1) +str(22)])
#            print(self.number_it)
#            print(sum(np.abs(resid.flatten())))
#            #return 
#            
#        return resid.flatten()     

#figGVD=plt.figure(figsize=(7,6))
#figGVD.add_subplot()
#pcolormesh(experiment.wavelength,experiment.x[:experiment.index2ps],pd.DataFrame(experiment.data).iloc[:experiment.index2ps].values,cmap='RdYlBu')
#plt.plot(experiment.wavelength,experiment.gvd)
#plt.axis([experiment.wavelength[0],experiment.wavelength[-1], experiment.x[0], experiment.x[experiment.index2ps-1]])
#plt.show(figGVD)
#
#c=experiment.gvd+experiment.point_pol_GVD[0]

#print polynom     
#coef=[2,-3,0,5]
#polynom=[f'{round(i,2)}X^{ii}' if ii>1 else f'{i}' for ii,i in enumerate(coef[::-1]) if i is not 0]
#polynom=('+'.join(polynom[::-1])).replace('+-','-')
        
#p_names=['t0_1','fwhm_1','tau1_1','tau2_1','tau3_1']
#std=[]
#for p in p_names:
#    std.append(experiment.resultados.params[p].stderr)
##    experiment.resultados.params[p].stderr = abs(experiment.resultados.params[p].value * 0.1)
#
#for p in experiment.resultados.params:
#    experiment.resultados.params[p].stderr = abs(experiment.resultados.params[p].value * 0.1)
#
#ci=lmfit.conf_interval(experiment, experiment.resultados,sigmas=[1],p_names=['t0_1','tau1_1','tau2_1','tau3_1'], trace=True)
#    
#fig, axes = plt.subplots(1, 1, figsize=(6, 4))
#cx_1, cy_1, grid = lmfit.conf_interval2d(experiment, experiment.resultados, 't0_1', 'tau1_1', 15, 15)
#ctp = axes.contourf(cx_1, cy_1, grid, np.linspace(0, 1, 11))
#fig.colorbar(ctp, ax=axes)
#axes.set_xlabel('t0_1')
#axes.set_ylabel('tau1_1')
#
#fig, axes = plt.subplots(1, 1, figsize=(6, 4))
#cx_2, cy_2, grid = lmfit.conf_interval2d(experiment, experiment.resultados, 'tau1_1', 'tau2_1', 15, 15)
#ctp = axes.contourf(cx_2, cy_2, grid, np.linspace(0, 1, 11))
#fig.colorbar(ctp, ax=axes)
#axes.set_xlabel('tau1_1')
#axes.set_ylabel('tau2_1')
#
#fig, axes = plt.subplots(1, 1, figsize=(6, 4))
#cx_2, cy_2, grid = lmfit.conf_interval2d(experiment, experiment.resultados, 'tau2_1', 'tau3_1', 10, 10)
#ctp = axes.contourf(cx_2, cy_2, grid, np.linspace(0, 1, 11))
#fig.colorbar(ctp, ax=axes)
#axes.set_xlabel('tau2_1')
#axes.set_ylabel('tau3_1')