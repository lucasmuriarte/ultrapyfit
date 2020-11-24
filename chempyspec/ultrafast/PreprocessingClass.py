# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:52:02 2020

@author: lucas
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter as SF
from ChirpCorrectionClass import ChripCorrection

class Preprocessing:
    
    @staticmethod
    def baselineSubstraction(data,nuber_spec=2,only_one=False): 
        if nuber_spec==0:
            only_one=True
        if only_one:
            mean=np.array(data[nuber_spec,:])
        else:
            if type(nuber_spec) is int:
                mean=np.mean(data[:nuber_spec,:],axis=0)
            elif type(nuber_spec) is list and len(nuber_spec)==2:
                mean=np.mean(data[nuber_spec[0]:nuber_spec[1],:],axis=0)
        for i in range(len(data)):
                data[i,:]=data[i,:]-mean
        return data
    
    @staticmethod
    def delPoints(points, data, dimension_vetor):
        ndata, nx = data.shape
        if dimension_vetor == None: dimension_vetor=np.array([i for i in range(ndata)])
        if type(points) is int or type(points) is float:
            points=[points]
        index=[np.argmin(abs(dimension_vetor-i)) for i in points]
        data=np.delete(data,index,axis=1)
        dimension_vetor=np.delete(dimension_vetor,index)
        if nx ==len(dimension_vetor):
            data=np.delete(data,index,axis=1)
        elif ndata ==len(dimension_vetor):
            data=np.delete(data,index,axis=0)
        else:
            raise 'The vector pass is not coincident with any of the data dimensions'
        return data, dimension_vetor
    
    @staticmethod
    def cutWavelenghts(data,wavelength,left=None,right=None,innercut=False):
        if wavelength is None:
            wavelength=np.array([i for i in range(len(data[1]))])
        statement=f'\t\tplease select only left or right, if booth an innercut will be done if innercut is set to True'
        if innercut==False and left is None and right is None:
            print(statement)
            return statement
        if innercut==False:
            if left is not None:
                assert right is None,statement
                cut_index=(pd.Series(wavelength)-left).abs().sort_values().index[0]
                if wavelength is not None:
                    wavelength=wavelength[:cut_index]
                else:
                    wavelength=wavelength[:cut_index]
                data=data[:,:cut_index]
            if right is not None:
                assert left is None,statement
                cut_index=(pd.Series(wavelength)-right).abs().sort_values().index[0]
                if wavelength is not None:
                    wavelength=wavelength[cut_index:]
                else:
                     wavelength=wavelength[cut_index:]
                data=data[:,cut_index:]
        elif innercut=='select':
            assert left is not None and right is not None, 'to select an area left and right margins should be given'
            cut_right=(pd.Series(wavelength)-right).abs().sort_values().index[0]
            cut_left=(pd.Series(wavelength)-left).abs().sort_values().index[0]
            if wavelength is not None:
                   wavelength = wavelength[cut_left:cut_right]
            else:
                    wavelength= wavelength[cut_left:cut_right]
            data=data[:,cut_left:cut_right]
        else:
            assert left is not None and right is not None, 'to do an inner cut left and right margins should be given'
            cut_right=(pd.Series(wavelength)-right).abs().sort_values().index[0]
            cut_left=(pd.Series(wavelength)-left).abs().sort_values().index[0]
            if wavelength is not None:
    #               wavelength[cut_left:cut_right]=wavelength[cut_left:cut_right]*0.0
                   wavelength=np.append(wavelength[:cut_left],wavelength[cut_right:])
            else:
    #                wavelength[cut_left:cut_right]=wavelength[cut_left:cut_right]*0.0
                    wavelength=np.append(wavelength[:cut_left], wavelength[cut_right:])
    #        data[:,cut_left:cut_right]=data[:,cut_left:cut_right]*0.0
            data=np.concatenate((data[:,:cut_left],data[:,cut_right:]),axis=1)
        return data, wavelength
    
    @staticmethod
    def cutTimes(data,time,mini=None,maxi=None):
        if mini is not None and maxi is None:
            min_index=np.argmin([abs(i-mini) for i in time])
            maxi_index=None
        elif maxi is not None and mini is None:
            maxi_index=np.argmin([abs(i-maxi) for i in time])
            min_index=0
        else:
            min_index=np.argmin([abs(i-mini) for i in time])
            maxi_index=np.argmin([abs(i-maxi) for i in time])
        if maxi_index is not None:
            time=time[min_index:maxi_index+1]
            data=data[min_index:maxi_index+1,:]
        else:
            time=time[min_index:]
            data=data[min_index:,:]
        return data, time
    
    @staticmethod
    def averageTimePoints(data,time,starting_point, step, method='log',grid_dense=5):
        point=np.argmin([abs(i-starting_point) for i in time])
        time_points=[i for i in time]
        value=step
        point=np.argmin([abs(i-starting_point) for i in time])
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
         
        data=data[:point+1,:]
        time=time[:point+1]
        for i in range(len(index)):
            column=np.mean(data[index[i][0]:index[i][-1]+1,:],axis=0).reshape(1,data.shape[1])
            timin=np.mean(time[index[i][0]:index[i][-1]+1])
            data=np.concatenate((data,column),axis=0)
            time=np.append(time,timin)
        return data, time
    
    @staticmethod
    def derivateSpace(data,window_length=25,polyorder=3,deriv=1,mode='mirror'):
        data2=0.0*data
        for i in range(len(data)):
            data2[i,:]=SF(data[i,:],window_length=window_length, polyorder=polyorder,deriv=deriv,mode=mode)
        return data2
    
    @staticmethod
    def shitTime(time,value):
        return time-value
    
    @staticmethod
    def substractPolynomBaseline(data,wavelength,points,order=3):
        assert len(points)>order, 'The number of points need to be higher than the polynom order'
        n_r,n_c=data.shape
        index=[np.argmin(abs(wavelength-i)) for i in points]
        data_corr=data*1.0
        for i in range(n_r):
            print(i)
            polynom=np.poly1d(np.polyfit(wavelength[index],data[i,index], order))
            data_corr[i,:]=data[i,:]-polynom(wavelength)
        
        return data_corr

    @staticmethod
    def correctChrip(data,wavelenght,time,method='selmeiller',return_details=False):
        if method == 'selmeiller':
            GVD=ChripCorrection(data,wavelenght,time)
            correct_data=GVD.GVDFromGrapth()
            details=f'\t\tCorrected with Sellmeier equation: {round(GVD.GVD_offset,2)} offset,\
            \n\t\tSiO2:{round(GVD.SiO2,2)} mm, \
            CaF2:{round(GVD.CaF2,2)} mm BK7:{round(GVD.BK7,2)} mm'
        elif method == 'polynom':
            pass #to be coded
        elif method == 'exponential':
            pass #to be coded
        if return_details:
            return correct_data, details
        else:    
            return correct_data