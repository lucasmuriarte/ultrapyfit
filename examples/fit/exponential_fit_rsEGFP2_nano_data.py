# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:18:19 2021

@author: lucas
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:19 2021

@author: lucas
"""
from ultrafast.fit.GlobalFit import GlobalFitExponential
from ultrafast.utils.divers import read_data, select_traces
from ultrafast.graphics.ExploreResults import ExploreResults
from ultrafast.graphics.ExploreData import ExploreData
from ultrafast.fit.GlobalParams import GlobExpParameters
import pandas as pd
import numpy as np

path='E:/PC/donnes/donnes nano/final nano files/Off to On/WT H2O/'
file_court=path +'ALL_340_WT_H2O 100 Ohm 20MHz temps court.abt'
file_long=path +'ALL_340_WT_H2O 560 Ohm 20MHz temps long.abt'
save=path

data_court=pd.read_csv(file_court,  skiprows=7,sep='\t').drop([0,10001,10002],axis=0)
data_long=pd.read_csv(file_long,  skiprows=7,sep='\t').drop([0,10001,10002],axis=0)
data_court=data_court.iloc[:-2,:]
data_long=data_long.iloc[:-2,:]

index=((data_long['Time[us]']-data_court['Time[us]'].values[-1]).abs()).sort_values().index[0]
value=10
for i in range(1,len(data_court.columns)):
    diff=(data_long.iloc[index:index+value,i]).mean()-(data_court.iloc[-value:-1,i]).mean()
    data_long.iloc[:,i]=data_long.iloc[:,i]-diff
data_all=data_court.append(data_long.iloc[index:,:])
data_all=data_all[data_all['Time[us]']>=0.2]
wave = np.linspace(340,520, 19)

time = data_all.pop('Time[us]')
# original_taus = [8, 30, 200]
params = GlobExpParameters(data_all.shape[1], [4, 40, 400])
params.adjustParams(0.1, False, None)
parameters = params.params

data_check  = ExploreData(time, data_all.values, wave)

fitter = GlobalFitExponential(time, data_all.values, 3, parameters, False,
                              wavelength=wave)
fitter.allow_stop = True
result = fitter.global_fit(maxfev=300)

explorer = ExploreResults(result)
explorer.print_results()
explorer.plot_fit()
explorer.plot_DAS()