# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:13:09 2020

@author: 79344
"""

import pandas as pd
import numpy as np
import re
from functools import wraps
    
def selectTraces(data,wavelength=None,space=10,points=1, avoid_regions=None):
    """ cut the data in time range
     Parameters
    ----------
    space: type=int or list or "auto": if type(int) a series of traces separated by the value indicated will be selected.
                                     if type(list) the traces in the list will be selected. 
                                     if auto, 10 sperated spectra will be selected
    points:bining point sourranding the selected wavelengths
    avoid_regions: list of sub_list 
        sub_list should have two elements defining the region to avoid in wavelenght values
        i. e. [[380,450],[520,530] traces with wavelength values between 380-450 and 520-530 will not be selected
    """
    dat=pd.DataFrame(data)
    wavelengths=pd.Series([float(i) for i in dat.columns])
    if space is 'auto':
       values=[i for i in range(len(wavelength))[::round(len(wavelength)/11)]] 
       values=values[1:]
    if type(space) is int:
        if wavelength is not None:
            wavelength_unit=1/((wavelength[-1]-wavelength[0])/len(wavelength))
            space=round(space*wavelength_unit)
        first=wavelengths.iloc[0+points]
        values=[first+space*i for i in range(len(wavelengths)) if first+space*i < wavelengths.iloc[-1]]
    elif type(space) is list:
        values=[np.argmin(abs(wavelength-i)) for i in space]
    selected_traces=[(wavelengths-values[i]).abs().sort_values().index[0] for i in range(len(values))]
    avoid_regions_index=[]
    if avoid_regions is not None: 
        assert type(avoid_regions) is list, 'Please regions should be indicated as a list'
        if type(avoid_regions[0]) is not list:
            avoid_regions=[avoid_regions]
        for i in avoid_regions:
            assert len(i) is 2,  'Please indicate 2 number to declare a region'
            i=sorted(i)
            avoid_wavelength=np.where((wavelength > i[0]) & (wavelength < i[1]))[0]
            if len(avoid_wavelength)>0: avoid_regions_index.append([avoid_wavelength[0],avoid_wavelength[-1]])
            selected_traces=[i for i in selected_traces if i not in avoid_wavelength]  
    if points == 0:
        dat=pd.DataFrame(data=[dat.iloc[:,i] for i in selected_traces],
                              columns=dat.index,index=[str(i+wavelengths[0]) for i in selected_traces]).transpose()
    else:
        if avoid_regions is not None:
            min_indexes=[]
            max_indexes=[]
            for trace in selected_traces:
                min_index=[sub_region[1] if trace-points > sub_region[0] and trace-points < sub_region[1] else trace-points for sub_region in avoid_regions_index]
                min_indexes.append(max(min_index))
                max_index=[sub_region[0] if trace+points > sub_region[0] and trace+points < sub_region[1] else trace+points for sub_region in avoid_regions_index]
                max_indexes.append(min(max_index))
            dat=pd.DataFrame(data=[dat.iloc[:,min_index:max_index].mean(axis=1) for min_index,max_index in zip(min_indexes,max_indexes)],
                              columns=dat.index,index=[str(i+wavelengths[0]) for i in selected_traces]).transpose()
    
    if wavelength is not None:
        wavelengths=pd.Series(wavelength)
        wavelength=np.array([wavelengths.iloc[i] for i in selected_traces])
    
    return dat.values, wavelength

def defineWeights(x,rango,typo='constant',val=5):
    '''typo should be a string exponential or r_expoential or exp_mix or constant
    exmaple:
    constant value 5, [1,1,1,1,...5,5,5,5,5,....1,1,1,1,1]
    exponential for val= 2 [1,1,1,1,....2,4,9,16,25,....,1,1,1,] 
                for val= 3 [1,1,1,1,....3,8,27,64,125,....,1,1,1,]
    r_expoential [1,1,1,1,...25,16,9,4,2,...1,1,1,]
    exp_mix [1,1,1,1,...2,4,9,4,2,...1,1,1,]'''
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
    return {'apply':False,'vector':np.array(weight),'type':typo,'range':rango,'value':val}


def readData(path,wavelength=0,time=0,wave_is_row=False,separator=',',decimal='.'):
    reader=ReadData()
    time, data, wavelenght = reader.readData(path,wavelength=wavelength,time=time
                                             ,wave_is_row=wave_is_row,separator=separator,decimal=decimal)   
    return time, data, wavelenght

class ReadData:
    def _readPandas(self,pandas):
        try:
            column=np.array([float(i) for i in pandas.columns.values])
        except:
            column=np.array([float((re.findall(r"[-+]?\d*\.\d*[eE]?[-+]?\d*|[-+]?\d+",i))[0]) for i in pandas.columns.values]).flatten()
        if type(pandas.index[0]) == str:
            row=np.array([float((re.findall(r"[-+]?\d*\.\d*[eE]?[-+]?\d*|[-+]?\d+",i))[0]) for i in pandas.index.values]).flatten()
        else:
            row=np.array([float(ii) for ii in pandas.index.values])
        return row, column
    
    def readData(self,path,wavelength=0,time=0,wave_is_row=True,separator=',',decimal='.'):     
        if wave_is_row:
            data_frame=pd.read_csv(path,sep=separator,index_col=wavelength,skiprows=time,decimal=decimal).dropna(how='all').dropna(how='all',axis=1)
            data_frame=data_frame.transpose().sort_index()
        else:
            data_frame=pd.read_csv(path,sep=separator,index_col=time,skiprows=wavelength,decimal=decimal).dropna(how='all').dropna(how='all',axis=1).sort_index()
        data_frame.fillna(0,inplace=True)
        wavelenght_dimension, time_dimension= self._readPandas(data_frame)
        return time_dimension, data_frame.transpose().values, wavelenght_dimension
    
def froze_it(cls):
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and hasattr(self, key):
            print("Class {} is frozen. Cannot modified {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls

@froze_it 
class UnvariableContainer:
    
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)

class VariableContainer:
    
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)
            
            
            
            
            
            
            
            
            