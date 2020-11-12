# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:05:16 2020

@author: lucas
"""

import pandas as pd
from scipy.signal import savgol_filter as SF
import matplotlib.pyplot as plt
import numpy as np
import copy  

class BasicSpectrum:
    """Contain basic functions that every spectrum should have
    
    atributes
    ----------
    
    x: array type
        independent variable of the spectrum
        
    y: array type
        independent variable of the spectrum
        
    data_table: pandas dataFrame
        data frame containig x and y with coulmn names x:'wavelengths' y:'absorbances'
    
    sort: bool
        if True sperctrum is order from the x minimum value to the x highest
    
    name: str
        name given to the spectrum
    
    name_x: str
         name of the x vector, i.e. 'Wavelngth (nm)'
    
    name_y: str
         name of the y vector, i.e. 'Absorbance'
    Methods
    -------
    
    """
    def __init__(self, x, y, sort=True,name=None,name_x=None,name_y=None):
        if name_x is None:
           self.name_x='x'
        else:
            self.name_x=name_x
        if name_y is None:
           self.name_y='y'
        else:
           self.name_y=name_y
        self.name=name
        self._xtranformed=False
        if sort:
            self.sort=True
            self.data_table=pd.DataFrame({self.name_x:x,self.name_y:y}).sort_values(by=[self.name_x]).reset_index(drop=True)  
        else:
            if x[0] < x[-1]:
                self.sort=True
            else:
                self.sort=False
            self.data_table=pd.DataFrame({self.name_x:x,self.name_y:y})
        self.x_original=self.data_table[self.name_x].values*1.0
        self.y_original=self.data_table[self.name_y].values*1.0
    
    def _minMaxIndex(self,low,high):
        if low == None:
                low=(self.data_table[self.name_x]).min()
        if high == None:
            high=(self.data_table[self.name_x]).max()
        high_index=(self.data_table[self.name_x]-high).abs().sort_values().index[0]
        low_index=(self.data_table[self.name_x]-low).abs().sort_values().index[0] - self.data_table.index[0]
        if low_index<high_index:
            return low_index,high_index
        elif high_index<low_index:
            return high_index,low_index
        else:
            raise 'low and high are same values'
    
    def transformXvector(self,func,new_name,undo=False):
        '''make a tranformation of x verto according to the function pass
        
         Paramaters
        ----------
        func: func(x):
            function that tranform x
            
        new_name: str
            string of the new x vector
            
        undo: bool
            if True and x has been transform, undo the tranformation, 
            No need to pass func or new_name if True
        
        '''
        if undo and self._xtranformed:
            self.data_table.drop(self.name_x,axis=1)
            self._xtranformed=False
            self.name_x=copy(self.old_x_name)
        elif self.wavenumber_calculated==False:
            self._xtranformed=True
            self.old_x_name=copy(self.name_x)
            self.name_x=new_name
            self.data_table[self.name_x]=self.data_table[self.old_x_name].apply(lambda x:func(x))
    
    def cut(self, low=None, high=None, itself=True):
        ''' cut the spectra acording to values of x 
        
        Paramaters
        ----------
        low: float
           minimum value of x
        
        high: float
           maximum value of x
        
        itself: bool
            if true cut the spectrum else returns a new spectrum object
        
        returns
        -------
        float, float
          x-position; y-maximum
        '''
        low_index,high_index=self._minMaxIndex(low,high)
        data_table= self.data_table[low_index:high_index]
        if itself:
            self.data_table= data_table
        else:
            BasicSpectrum(data_table[self.name_x],data_table[self.name_y],self.name,self.name_x,self.name_y)
    
    def __sub__(self, obj):
        '''subtract two spectra'''
        ret_obj = BasicSpectrum(self.data_table[self.name_x],self.data_table[self.name_y],self.name,self.name_x,self.name_y)
        ret_obj.data_table[ret_obj.name_y]= self.data_table[self.name_y] - obj.data_table[obj.name_y]
        return ret_obj
    
    def __mul__(self, number):
        '''multiply spectrum by number'''
        ret_obj = BasicSpectrum(self.data_table[self.name_x],self.data_table[self.name_y],self.name,self.name_x,self.name_y)
        ret_obj.data_table['absorbances']= self.data_table['absorbances'] * number
        ret_obj.y=self.y*number
        return ret_obj
    
    def __add__(self, obj):
        '''add two spectra'''
        ret_obj = BasicSpectrum(self.data_table[self.name_x],self.data_table[self.name_y],self.name,self.name_x,self.name_y)
        ret_obj.data_table[ret_obj.name_y]= self.data_table[self.name_y] + obj.data_table[obj.name_y]
        return ret_obj
    
    def __truediv__(self, number):
        '''divide spectrum by a number'''
        ret_obj = BasicSpectrum(self.data_table[self.name_x],self.data_table[self.name_y],self.name,self.name_x,self.name_y)
        ret_obj.data_table['absorbances']= self.data_table['absorbances'] / number
        ret_obj.y=self.y/number
        return ret_obj
        
    def obtainValue(self,value):
        ''' returns the y value of the correspoding x value
        
        Paramaters
        ----------
        valu: float
            value of x
        
        returns
        -------
        float
           y value
        '''
        index_value=(self.data_table[self.name_x]-value).abs().sort_values().index[0]
        return self.data_table[self.name_y][index_value]
            
    def obtainMax(self,low=None,high=None,Print=False):
        ''' returns the maximum of the y vecto and the correspodint x postion
        
        Paramaters
        ----------
        low: float
           minimum value of x range
           
        high: float
           maximum value of x range
           
        Print: bool
            print the results out
        
        returns
        -------
        float, float
          x-position; y-maximum
        '''
        low_index,high_index=self._minMaxIndex(low,high)
        maxi_index=self.data_table[self.name_y][low_index:high_index].idxmax()
        if Print:
            print('the maximum is at: '+ str(self.data_table[self.name_x][maxi_index])\
                  +'; And the value is: '+str(self.data_table[self.name_y][maxi_index]))
        return self.data_table[self.name_x][maxi_index],self.data_table[self.name_y][maxi_index]
        
    def obtainMin(self,low=None,high=None,Print=False):
        ''' returns the minimum of the y vector and the correspodint x postion
        Paramaters
        ----------
        low: float
           minimum value of x range
           
        high: float
           maximum value of x range
        
        Print: bool
            print the results out
        
        returns
        -------
        float, float
          x-position; y-minimum
        '''
        low_index,high_index=self._minMaxIndex(low,high)
        min_index=self.data_table[self.name_y][low_index:high_index].idxmin()
        if Print:
            print('the minimum is at: '+ str(self.data_table[self.name_x][min_index])\
                  +'; And the value is: '+str(self.data_table[self.name_y][min_index]))
        return self.data_table[self.name_x][min_index],self.data_table[self.name_y][min_index]
    
    def forcePositive(self,itself=True):
        '''add minimum value of the y vector to itself and thus force it to be positive'''
        self.positive=True
        mini=self.data_table[self.name_x].min()
        if mini<0:
            if itself:
                self.data_table[self.name_x] = self.data_table[self.name_y]-mini   
            else:
               spec= BasicSpectrum(self.data_table[self.name_x],self.data_table[self.name_y]-mini,self.name,self.name_x,self.name_y)
               return  spec 
    
    def normalize(self,wavelength=None,low=None,high=None,itself=True):
        ''' normalize the spectra so that the wavelenght value is 1.
        If wavelenght is None the maximum of the range between low and 
        high values will be use to normalize the spectrum.
        
        Paramaters
        ----------
        wavelength: float
            wavelenth value where the spectra shoudl be 1
            if None the maximum will be chose
        
        low: float
           minimum value of x range
           
        high: float
           maximum value of x range
           
        itself: bool
            if true normalize the spectrum else returns a new spectrum object
         '''
        if wavelength is None:
             _,val=self.obtainMax(low,high)
        else:
           val=self.obtainValue(wavelength)
        data=self.data_table.copy()
        data[self.name_y]=self.data[self.name_y]/val
        if itself:
            self.data_table[self.name_y]=data[self.name_y]
        else:
            spec= BasicSpectrum(data[self.name_x],data[self.name_y],self.name,self.name_x,self.name_y)
            return  spec
                   
          
       
    def plotSpectrum(self,axes='current',x_label=None, y_label=None,\
             label=None, fill=False, two_Xaxes=True,  **kwargs):
        '''Plot the spectrum 
        
        Paramaters
        ----------
        axes: str or matplotlib axes object
              -  "current" plots in the last open axes
              -  "new" create a new axes
              -  pass a matplotlib axes where the spectra will be plot 
              
        x_label: str
            the x label of the plot
        
        y_label: str
            the y label of the plot
            
        label: str
            the legend to be display (pass "_" form no label)
            
        fill: bool
            fill area under the spectrum curve
            
        two_Xaxes: bool
            in case two x is tranformed (i.e. wavelenght and wavenumber) 
            and if True, the old and new axis will be plot 
                
        kwargs: dict
            matplot lib kwargs for plt.plot()
        
        extra_kwargs: dict
            this parameters can be pases as kwargs eve if they are not from plt.plot()
                fsize: 14
                small_tick: True
                which:both
                direction:in
                
        returns
        -------
            matplotlib axex object
        '''
        plot_properties=dict({'fsize':14,'small_tick':True,'which':'both','direction':'in'},**kwargs)
        for i in plot_properties.keys():
            if i in kwargs.keys():  kwargs.pop(i) 
        if x_label == None:
            x_label=self.name_x
        if y_label == None:
            y_label=self.name_y
        if axes =='current':
            ax1=plt.gca()
            if type(ax1) == str:
              axes='new'  
        if axes == 'new':
            fig, ax1 = plt.subplots(1)
        else:
            ax1=axes
        if label is None:
            ax1.plot(self.data_table[self.name_x],self.data_table[self.name_y],**kwargs)
        else:
            if fill: 
                ax1.plot(self.data_table[self.name_x],self.data_table[self.name_y],**kwargs)
                plt.fill_between(self.data_table[self.name_x],self.data_table[self.name_y],color=plt.gca().lines[-1].get_color(),alpha=0.5,label=label)
            else:
                ax1.plot(self.data_table[self.name_x],self.data_table[self.name_y],**kwargs,label=label,)
        ax1.set_xlabel(x_label,size=plot_properties['fsize'])
        ax1.set_ylabel( y_label,size=plot_properties['fsize'])
        plt.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
        ax1.axes.tick_params(which='both',direction='in',top=True,right=True,labelsize=plot_properties['fsize'])
        if plot_properties['small_tick']:
            ax1.minorticks_on()
        if self.name != None:
            plt.legend(self.name,prop={'size': plot_properties['fsize']})
        else:
            if label is not None:
                plt.legend(prop={'size': plot_properties['fsize']})
        plt.tick_params(axis = 'both', which = 'major', labelsize = plot_properties['fsize'])
        plt.xlim(self.data_table[self.name_x][self.data_table.index[0]],self.data_table[self.name_x][self.data_table.index[-1]])
        if two_Xaxes and self._xtranformed:
            self._twinyAxes(plot_properties['fsize'])
        return ax1
        
    def _twinyAxes(self,fsize):
        """Twin X axes for plotSpectrum function"""
        ax1=plt.gca()
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2=ax1.twiny()
        mini=int(np.min(self.x_wave))
        maxi=int(np.max(self.x_wave))
        labels=[i for i in range(mini,maxi,int(abs(mini-maxi)/5))]
        ax2.set_xlim(ax1.get_xlim())
        new_tick_locations=[10**7/float(i) for i in labels]
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(labels)
        ax2.set_xlabel('Wavelength (nm)',size=fsize)
        plt.tick_params(axis = 'both', which = 'major', labelsize = fsize)
        
    def calculateArea(self,low=None,high=None,Print=False):
        ''' cut the spectra acording to values of x 
        
        Paramaters
        ----------
        low: float
           minimum value of x
           
        high: float
           maximum value of x
           
        Print: bool
            print the results out
        
        returns: float 
        -------
            the are under the curve in the range
        '''
        low_index,high_index=self._minMaxIndex(low,high)
        y=self.data_table[self.name_y][low_index:high_index].values
        x=self.data_table[self.name_x][low_index:high_index].values
        area=np.trapz(y,x)
        if Print:
            print(f'the area in the range {low}-{high} is: {area}')
        return area
    
    def averageOfRange(self,low=None,high=None,Print=False):
        ''' Returns the average value of "y" from the given "x" range 
        
        Paramaters
        ----------
        low: float
           minimum value of x
        
        high: float
           maximum value of x
           
        Print: bool
            print the results out
        
        returns: float
        -------
            the average the range
        '''
        low_index,high_index=self._minMaxIndex(low,high)
        mean=(self.data_table[self.name_y][low_index:high_index]).mean()
        if Print:
            print(f'the mean unde the range {low}-{high} nm is: {mean}')
        return mean
        
    def baselineCorrection(self,low=None,high=None, itself=True):
        ''' subrtact the average value of a range to the spectra acording to values of x 
        
        Paramaters
        ----------
        low: float
           minimum value of x range
        
        high: float
           maximum value of x range
        
        itself: bool
            if true subtract the spectrum else returns a new spectrum object
        
        returns
        -------
        float, float
          x-position; y-maximum
        '''
        low_index,high_index=self._minMaxIndex(low,high)
        data=self.data_table.copy()
        mean=self.averageOfRange(low,high)
#        mean=(data[self.name_y][(data[self.name_x] >= low) \
#              & (data[self.name_x] <= high)]).mean()
        data[self.name_y] = data[self.name_y]-mean 
        if itself:
            self.data_table[self.name_y]=data[self.name_y]
        else:
            spec= BasicSpectrum(data[self.name_x],data[self.name_y],self.name,self.name_x,self.name_y)
            return  spec
                                                                                                                                
       
            
            