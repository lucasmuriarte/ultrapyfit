# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:52:02 2020

@author: lucas
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter as SF
from chempyspec.ultrafast.ChirpCorrectionClass import ChripCorrection


class ExperimentException(Exception):
    """General Purpose Exception."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return "{}".format(self.msg)
    
        
class Preprocessing:
    """
    Class containing main preprocessing functions as static methods
    
    All methods assumes tha the data array pass has: 1) data is a 2darray that 
    has no wavelength or time data points, time and wavelength are given as separate
    1d arrays. 2) In the original data set the number of rows shape is the time
    points and the number of columns the wavelength points. (Some functions work
    independently of the orientation of the data)
    
    The transient data pass should be: 
        data 2d_arrays (time point rows and wavelength columns)
        time 1d_arrays 
        wavelength 1d_arrays
        e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
    """
    
    @staticmethod
    def baselineSubstraction(data, number_spec=2, only_one=False):
        """
        Subtract a initial spectrum or an average of initial spectra
        (normally before time = 0) to the entire data set.
        
        
        Parameters
        ----------
        data: ndarray
          Original data set where the number of rows should be the time points
          and the number of columns the wavelength points
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        number_spec: int or list
            This parameter defines the number of spectra to be subtracted.
            If int should be the index of the spectrum to be subtracted
            If list should be of length 2 and contain initial and final spectra
            index that want to be subtracted
        
        only_one: bool (default False)
            if True only the spectrum (at index number_spec) is subtracted
            if False an average from the spectrum 0 to number_spec is subtracted 
            (only applicable if number_spec is an int)
        
        e.g.1: number_spec = [2,5] an average from spectra 2 to 5 is subtracted
                (where 2 and 5 are included)
        
        e.g.2: number_spec = 5 an average from spectra 0 to 5 is subtracted
                if only_one = False; if not only spectrum 5 is subtracted
        
        Returns
        ----------
        ndarray of size equal data 
        """
        data_return = data*0.0
        if number_spec == 0:
            only_one = True
        if only_one and type(number_spec) is int:
            mean = np.array(data[number_spec, :])
        else:
            if type(number_spec) is int:
                mean = np.mean(data[:number_spec, :], axis=0)
            elif type(number_spec) is list and len(number_spec) == 2:
                mean = np.mean(data[number_spec[0]:number_spec[1]+1, :], axis=0)
        for i in range(len(data)):
            data_return[i, :] = data[i, :] - mean
        return data_return
    
    @staticmethod
    def delPoints(points, data, dimension_vector, axis=0):
        """
        Delete rows or columns from the data set according to the closest values
        given in points to the values found in dimension_vector. The length of
        dimension_vector should be equivalent to one of the data dimensions. 
        
        Notice that the function will automatically detect the dimension of the
        delete rows or columns in data if any of their dimensions is equal to the
        length of the dimension_vector. In case that both dimensions are the same
        the axis should be given, by default is 0, which is equivalent to the
        time dimension.
        
        i.e.: 
        points = [3,5] 
        dimension_vector = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6]
        len(dimension_vector) >>> 8
        data.shape >>> (8, 10)
        
        returns
            return_data: data array where rows 3 and 6 are deleted
            return_data.shape >>> (6, 10)
            return_dimension_vector >>> [0.7, 1.4, 2.1, 3.5, 4.2, 5.6]
        
        Parameters
        ----------
        points: int, list or None
          estimate values of time, the closes values of dimension_vector to the points given
          will be deleted
        
        data: ndarray
          Original data set where the number of rows should be the time points
          and the number of columns the wavelength points
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        dimension_vector: 1darray
            wavelength or time vectors where points should be deleted
            
        axis: int, 0 or 1 (default: None)
            indicates the axis where to cut in case data has identical number 
            of rows and columns. (not needed in other cases)            
            
        
        e.g.1: number_spec = [2,5] an average from spectra 2 to 5 is subtracted
        
        e.g.2: number_spec = 5 an average from spectra 0 to 5 is subtracted
                if only_one = False; if not only spectrum 5 is subtracted
        
        Returns
        ----------
        initial data ndarrays with rows or columns deleted and dimension_vector
        with points deleted.
        """
        ndata, nx = data.shape
        if dimension_vector is None:
            dimension_vector = np.array([i for i in range(ndata)])
        if type(points) is int or type(points) is float:
            points = [points]
        index = [np.argmin(abs(dimension_vector - i)) for i in points]
        if ndata == nx:
            if axis not in [0, 1]:
                exception = 'axis should be 0 (time) or 1 (wavelength)'
                raise ExperimentException(exception)
            data_return = np.delete(data, index, axis=axis)
        elif ndata == len(dimension_vector):
            data_return = np.delete(data, index, axis=0)
        elif nx == len(dimension_vector):
            data_return = np.delete(data, index, axis=1)
        else:
            exception = 'The length of the vector pass is not coincident with \
                        any of the data dimensions'
            print('exception')
            raise ExperimentException(exception)
        dimension_vector = np.delete(dimension_vector, index)
        return data_return, dimension_vector
    
    @staticmethod
    def cutColumns(data, columns, mini=None, maxi=None, innerdata=None):
        """
        Cut columns of the data set and wavelength vector according to the closest
        values of mini and maxi margins given.  
        (The function assumes column vector is sorted from low to high values)
        
        Parameters
        ----------     
        data: ndarray
          Normally in  the original data set the number of rows should be the 
          time points and the number of columns the Columns points. Although 
          this function works with transposes data sets
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        columns: 1darray or None
            Wavelength vectors where regions are cut.
            If None an array from 0 to data number of columns is created and
            returned after cutting. The parameters mini and maxi should be given
            according to indexes
            
        mini: int, float or None (default: None)
          data higher than this value is kept
        
        maxi: int, float or None (default: None)
          data lower than this value is kept
          
        innerdata: cut or select (default: None)
            Only need if both mini and right maxi are given
            indicates if data inside the mini and maxi limits should be cut or
            selected.
        
        Returns
        ----------
        initial data and columns vector cut in the areas indicated
        """
        
        if columns is None:
            columns_res = np.array([i for i in range(len(data[1]))])
        if len(columns) != data.shape[1]:
            statement_1 = 'The size of the columns vector is not equivalent with\
                    the number of columns of data'
            raise ExperimentException(statement_1)
        if innerdata is not None:
            statement_3 = 'to select or cut data mini and maxi need to be given'
            if mini is None:
                raise ExperimentException(statement_3)
            if maxi is None:
                raise ExperimentException(statement_3)
        if mini is None and maxi is None:
            statement_2 = 'please indicate only mini or maxi margins, or booth \
            if data inside margins want to be cut or selected with innerdata'
            raise ExperimentException(statement_2)
        elif maxi is not None and mini is None:
            cut_index = (pd.Series(columns)-maxi).abs().sort_values().index[0]
            if columns[cut_index] < maxi:
                cut_index += 1
            columns_res = columns[:cut_index]
            data_res = data[:, :cut_index]
        elif mini is not None and maxi is None:
            cut_index = (pd.Series(columns)-mini).abs().sort_values().index[0]
            if columns[cut_index] < mini:
                cut_index += 1
            columns_res = columns[cut_index:]
            data_res = data[:, cut_index:]
        elif innerdata == 'select':
            cut_maxi = (pd.Series(columns)-maxi).abs().sort_values().index[0]
            cut_mini = (pd.Series(columns)-mini).abs().sort_values().index[0]
            if columns[cut_mini] < mini:
                cut_mini += 1
            if columns[cut_maxi] < maxi:
                cut_maxi += 1
            columns_res = columns[cut_mini:cut_maxi]
            data_res = data[:, cut_mini:cut_maxi]
        elif innerdata == 'cut':
            cut_maxi = (pd.Series(columns)-maxi).abs().sort_values().index[0]
            cut_mini = (pd.Series(columns)-mini).abs().sort_values().index[0]
            if columns[cut_mini] < mini:
                cut_mini += 1
            if columns[cut_maxi] < maxi:
                cut_maxi += 1
            columns_res = np.append(columns[:cut_mini], columns[cut_maxi:])
            data_res = np.concatenate((data[:, :cut_mini], data[:, cut_maxi:]), axis=1)
        else:
            statement_4 = 'if mini and maxi margins are be given indicates \
            that innerdata action either cut or select'
            raise ExperimentException(statement_4)
        return data_res, columns_res
    
    @staticmethod
    def cutRows(data, rows, mini=None, maxi=None):
        """
        Cut rows of the data set and time vector according to the closest
        values of mini and maxi margins given. Contrary to cutColumns functions,
        cut are not available since in time resolved spectroscopy is not logical
        to cut a complete area of recorded times. Therefore, giving mini and maxi
        margins will result in selection of inner time values.
        (The function assumes rows vector is sorted from low to high values)
        
        Parameters
        ----------     
        data: ndarray
          Normally in  the original data set the number of rows should be the 
          time points and the number of columns the Columns points. Although 
          this function works with transposes data sets
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        rows: 1darray or
            Normally time vector where regions are cut.
            
        mini: int, float or None (default: None)
          data higher than this value is kept
        
        maxi: int, float or None (default: None)
          data lower than this value is kept
        
        Returns
        ----------
        initial data and rows vector cut in the areas indicated
        """
        if mini is not None and maxi is None:
            min_index = np.argmin([abs(i-mini) for i in rows])
            maxi_index = None
        elif maxi is not None and mini is None:
            maxi_index = np.argmin([abs(i-maxi) for i in rows])
            min_index = 0
        else:
            min_index = np.argmin([abs(i-mini) for i in rows])
            maxi_index = np.argmin([abs(i-maxi) for i in rows])
        if maxi_index is not None:
            rows_res = rows[min_index:maxi_index+1]
            data_res = data[min_index:maxi_index+1, :]
        else:
            rows_res = rows[min_index:]
            data_res = data[min_index:, :]
        return data_res, rows_res
    
    @staticmethod
    def averageTimePoints(data, time, starting_point, step, 
                          method='log', grid_dense=5):
        """
        Average time points collected (rows). This function can be use to average 
        time points. Useful in multiprobe time-resolved experiments or flash-
        photolysis experiments recorded with a Photo multiplier tube where the number
        of time points is very long and are equally spaced.
        (The function assumes time vector is sorted from low to high values)
        
        
        Parameters
        ----------     
        data: ndarray
          Normally in  the original data set the number of rows should be the 
          time points and the number of columns the Columns points. Although 
          this function works with transposes data sets
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        time: 1darray or 
            time vector 
            
        starting_point: int or float 
          time points higher than this the function will be applied
        
        step: int, float or None
          step to consider for averaging data points
        
        method: 'log' or 'constant' (default: 'log')
            If constant: after starting_point the the function will return average 
            time points between the step.
              
            If log the firsts step is step/grid_dense and the following points
            are (step/grid_dense)*n where n is point number
        
        grid_dense: int or float higher than 1 (default: 5)
            density of the log grid that will be applied. To high values will not
            have effect if: start_point + step/grid_dense is lower than the
            difference between the first two consecutive points higher than
            start_point. The higher the value the higher the grid dense will be.
            return.
            (only applicable if method is 'log')
        
        e.g.:
            time [1,2,3,4 .... 70,71,72,73,74]
            step = 10
            starting_point = 5
            method 'constant'
                time points return are [1,2,3,4,5,10,20,30,40...]
            method 'log'
                time points return are [1,2,3,4,5,6,9,14,21,30,41,54,67.5]
            
        Returns
        ----------
        initial data and time vector averaged
        """
        if method not in ['log', 'constant']:
            statement = 'Method should be "log" or "constant"'
            raise ExperimentException(statement)
        point = np.argmin([abs(i-starting_point) for i in time])
        time_points = [i for i in time]
        value = step
        index = []
        it = point
        log = 1
        if method == 'log':
            if grid_dense <= 1:
                statement = 'grid_dense should be higher than 1'
                raise ExperimentException(statement)
            value /= grid_dense
        number = time_points[point] + value
        while number < time_points[-1]:  
            time_average = [i for i in range(it+1, len(time_points)) if time_points[i] <= number]
            if method == 'log':
                log += 1
            number += value*log
            if len(time_average) >= 1:
                index.append(time_average)
                it = time_average[-1]
            else:
                log += 1
        index.append([i for i in range(index[-1][-1]+1, len(time_points))])
         
        data_res = data[:point+1, :]
        time_res = time[:point+1]
        for i in range(len(index)):
            if method == 'log' and len(index[i]) == 1:
                column = (data[index[i][0], :]).reshape(1, data.shape[1])
                timin = time[index[i][0]]
            else:
                column = np.mean(data[index[i][0]:index[i][-1], :],
                                 axis=0).reshape(1, data.shape[1])
                timin = np.mean(time[index[i][0]:index[i][-1]])
            data_res = np.concatenate((data_res, column), axis=0)
            time_res = np.append(time_res, timin)
        return data_res, time_res
    
    @staticmethod
    def derivateSpace(data, window_length=25, polyorder=3, deriv=1, mode='mirror'):
        """
        Apply a Savitky-Golay filter to the data in the spectral range (rows).
        This function can be used to correct for baseline fluctuations and still 
        perform a global fit or a single fit to obtain the decay times.  Uses 
        scipy.signal.savgol_filter (check scipy documentation for more information)
        
        
        Parameters
        ----------     
        data: ndarray
          Normally in  the original data set the number of rows should be the 
          time points and the number of columns the Columns points. Although 
          this function works with transposes data sets
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        window_length: odd int value (default: 25)
            length defining the points for polynomial fitting
            
        polyorder: int or float (default: 3)
          order of the polynomial fit
        
        deriv: int, float or None (default: 1)
          order of the derivative after fitting
        
        mode: (default: 'mirror')
            mode to evaluate bounders after derivation, check scipy.signal.savgol_filter
            for the different options
            
        Returns
        ----------
        initial data and time vector averaged
        """
        data2 = 0.0*data
        for i in range(len(data)):
            data2[i, :] = SF(data[i, :], window_length=window_length,
                             polyorder=polyorder, deriv=deriv, mode=mode)
        return data2
    
    @staticmethod
    def shitTime(time, value):
        """
        Shift the time vector by a value
        
        
        Parameters
        ----------     
        time: 1darray
          time vector
        
        value: int or float 
            value shifting the time vector
        
        Returns
        ----------
        time value shifted
        """    
        return time - value
    
    @staticmethod
    def substractPolynomBaseline(data, wavelength, points, order=3):
        """
        Fit and subtract a polynomial to the data in the spectral range (rows).
        This function can be used to correct for baseline fluctuations typically
        found in time resolved IR spectroscopy.
        
        Parameters
        ----------     
        data: ndarray
          Normally in  the original data set the number of rows should be the 
          time points and the number of columns the Columns points. Although 
          this function works with transposes data sets
              e.g.: data.shape = (50,250)
                data has 250 wavelength points and 50 time points
        
        wavelength: 1darray
            wavelength vector
       
        points: list
            list containing the wavelength values where the different transient
            spectra should be zero
       
        order: int or float (default: 3)
           order of the polynomial fit
        
            
        Returns
        ----------
        initial data and time vector averaged
        """
        if len(points) > order:
            statement = 'The number of points need to be higher than the polynomial order'
            raise ExperimentException(statement)
        n_r, n_c = data.shape
        index = [np.argmin(abs(wavelength-i)) for i in points]
        data_corr = data*1.0
        for i in range(n_r):
            print(i)
            polynom = np.poly1d(np.polyfit(wavelength[index], data[i, index], order))
            data_corr[i, :] = data[i, :] - polynom(wavelength)
        return data_corr

    @staticmethod
    def correctChrip(data, wavelength, time, method='selmeiller', return_details=False):
        if method == 'selmeiller':
            GVD = ChripCorrection(data, wavelength, time)
            correct_data = GVD.GVDFromGrapth()
            details = f'\t\tCorrected with selmeiller equation: {round(GVD.GVD_offset,2)} offset,\
            \n\t\tSiO2:{round(GVD.SiO2,2)} mm, \
            CaF2:{round(GVD.CaF2,2)} mm BK7:{round(GVD.BK7,2)} mm'
        elif method == 'polynomial':
            # to be coded
            pass
        elif method == 'exponential':
            # to be coded
            pass
        if return_details:
            return correct_data, details
        else:    
            return correct_data
