# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:40:36 2020

@author: 79344
"""
from collections import OrderedDict 
import pandas as pd
from snapToCursorClass import SnaptoCursor
from BasicSpectrumClass.py import BasicSpectrum
 
class SeriesSpectra(OrderedDict):
    def __init__(self, z_dimension=None):
        self.z_dimension = z_dimension
        self.maximas = []
        self.minimas = []
        self.areas_at_wavelenght={}
        self.data_frame = pd.DataFrame()
        self.x_label=None
        self.y_label=None
        self.z_label=None
        self.share_x = True
        self.current_x=None
        self.cursor=SnaptoCursor()
        self.adapt_spectrum_wiht_diffent_x_method='interpolate'
        super().__init__()
    
    def add(self,name,x,y,sort=True,adapt_x=False):
        '''add spectrum to the series
        
        Parameters
        -------------------------------------------
        name (str): name of the spectrum
        x: wavelenght values of the spectrum
        y: measured propertie values
        sort (bool) default True. If True the spectrum will be sorted from low to high according to x values
        new_current_x (bool) default False. If True the the spectra will be adapted series current_x values
        '''
        self.add[name]=BasicSpectrum(x,y,sort,name)

    def loadData(self,file,x=0,x_vertical=True,**kwargs):
        '''load data into spectral serie
            
            Parameters
            -------------------------------------------
            file (str): path for reading the series 
            x (int): postion of the x vector inside the matrix
            x_vertical (bool): if vertical x vector is a column else x vector is a row
            '''  
        pass
    
    def rename(self, mapper, itself=True):
        '''rename the spectra of the series
    
        Parameters
        -------------------------------------------
        mapper :  dict-like containing the olde and new names
        itself (bool): default True. If True rename the series else returns a spectral series with new names'''
        pass
    
    def getKeyNumber(self,number):
        '''retunr the key of an elelment position
        returns: key of element at position number
        
        Parameters
        -------------------------------------------
        number (int): elelment position'''
        
        key=[i for i in self.keys][number]
        return key
    
    def baselineSubtraction(self,low=None,high=None,aplly_to='all',itself=False):
        '''substract a baseline calculating the average of the point between low and high
            
        Parameters
        -------------------------------------------
        file (str): path for reading the series 
        x (int): postion of the x vector inside the matrix
        aplly : if 'all' apply to all the Series of spectra. if a number or a key to an specific spectrum, 
                if a list with keys or number to all spectra in the list
        itself (bool): default False. If True apply to the series else returns the series
        '''
        pass
    
    def difference(self):
        pass
    
    def __sub__(self, series):
        '''subtratc a series of identical x. 
        If the series have diferent lenght the returning series 
        has the size of the smallest of the two sereis 
        
        returns: and object series
        
        Parameters
        -------------------------------------------
        series: series to subtract'''
        
        return series
        pass

    def __add__(self, series):
        '''Add a series of identical x. 
        If the series have diferent lenght the returning series 
        has the size of the smallest of the two sereis 
        
        returns: and object series
        
        Parameters
        -------------------------------------------
        series: series to subtract'''
        
        return series
        pass
        
    def apply(self,func,aplly_to='all',itself=False):
        '''Apply a function to the Y values of spectra in the series
            
        Parameters
        -------------------------------------------
        func : function to apply 
        aplly : if 'all' apply to all the Series of spectra. if a number or a key to an specific spectrum, 
                if a list with keys or number to all spectra in the list
        itself (bool): default False. If True apply to the series else returns the series
        '''
        pass
    
    def subtractOneToAll(self,name,aplly_to='all', itself=False ):
        '''Subctract one spectrum to the rest of spectra in the series
            
        Parameters
        -------------------------------------------
        name (int/str): key or spectrum number to be substracte to the spectra series
        aplly : if 'all' apply to all the Series of spectra. if a number or a key to an specific spectrum, 
                if a list with keys or numbers to all spectra in the list
        itself (bool): default False. If True apply to the series else returns the series
        '''
        pass
    
    def getAverage(self,spectra,append=False,name=None):
        '''Sum one spectrum to the rest of spectra in the series
         
        return BasicSpectrum object containing the averige of spectra
            
        Parameters
        -------------------------------------------
        spectra (list): list conatining key or spectrum number to be substracte to the spectra series
        append (bool): default Flase. if True the spectra is appended to the series, 
        name (str): default None. name of the average spectra
        '''
        pass
    
    def addOneToAll(self, name,aplly_to='all', itself=False):
        '''Sum one spectrum to the rest of spectra in the series
            
        Parameters
        -------------------------------------------
        name (int/str): key or spectrum number to be substracte to the spectra series
        aplly : if 'all' apply to all the Series of spectra. if a number or a key to an specific spectrum, 
                if a list with keys or numbers to all spectra in the list
        itself (bool): default False. If True apply to the series else returns the series
        '''
        pass
    
        
    def addConstantAll(self, value,aplly_to='all', itself=False):
        '''add aconstant value to the spectra in the series
            
        Parameters
        -------------------------------------------
        value (int/float): value to be added
        aplly : if 'all' apply to all the Series of spectra. if a number or a key to an specific spectrum, 
                if a list with keys or numbers to all spectra in the list
        itself (bool): default False. If True apply to the series else returns the series
        '''
        pass
    
    def plotSeries(self):
         '''Plot the spectral series
            
        Parameters
         -------------------------------------------
         to be defined'''
         pass
     
    def plotMaximax(self):
       '''Plot the maximas of the spectral series'''
       pass
        
    def plotMinimas(self):
        '''Plot the minimas of the spectral series'''
        pass
     
    def decomposeInSumOfSpectra(self):
        '''Decompose the series in a sum of spectra'''
        pass
    
    def normalize(self,by='max'):
        '''normalize all spectra of  the series''' 
    
    def smooth(self):
        '''smooth all spectra of  the series''' 
    