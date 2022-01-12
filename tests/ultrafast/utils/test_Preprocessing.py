# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:00:33 2021

@author: lucas
"""

import unittest
import numpy as np
from ultrapyfit.utils.Preprocessing import Preprocessing, ExperimentException
from parameterized import parameterized

data_wave = np.ones((75, 150))
for i in range(data_wave.shape[0]):
    data_wave[i, :] = data_wave[i, :]*i
    
data_time = np.ones((75, 150))
for i in range(data_time.shape[1]):
    data_time[:, i] = data_time[:, i]*i
wave = np.linspace(351, 350+50, 150)
time = np.linspace(0, 49, 75)


class TestPreprocessing(unittest.TestCase):
    """
    test for Preprocessing Class
    """
    
    def assertEqualArray(self, array1, array2):
        """
        returns "True" if all elements of two arrays are identical
        """
        value = (array1 == array2).all()
        return value
    
    def assertNearlyEqualArray(self, array1, array2, decimal):
        """ 
        returns "True" if all elements of two arrays 
        are identical until decimal
        """
        dif = np.array(array1) - np.array(array2)
        value = (dif < 10**(-decimal)).all()
        return value
        
    @parameterized.expand([[0, True],
                           [0, False],
                           [[3, 5], False],
                           [[2, 5], False],
                           [3, False],
                           [5, False],
                           [2, False],
                           [1, True]])
    def test_baselineSubstraction(self, number_spec, only_one):
        """
        test for baseline sbtarction considering all rows of data_wave have only
        one number increassing from 0 to 74
        """
        result = Preprocessing.baseline_substraction(data_wave, number_spec, only_one)
        exp = data_wave*0.0
        if only_one or number_spec == 0:
            resta = np.ones((1, 150))*number_spec
        else:
            if type(number_spec) == int:
                number_spec = [0, number_spec-1]
            resta = np.ones((1, 150))*np.mean(number_spec)
        for i in range(data_wave.shape[0]):
            exp[i, :] = data_wave[i, :]-resta
        self.assertTrue(self.assertNearlyEqualArray(result, exp, 25))
    
    @parameterized.expand([[[0, 1, 25, 75]],
                           [[0, 25, 15, 75]],
                           [25],
                           [0],
                           [75],
                           [[350, 375, 365, 400]],
                           [375],
                           [351],
                           [400]])
    def test_delPoints(self, points):
        if type(points) == int:
            points = [points]
        if points[0] < 349:
            data_res, vec_res = Preprocessing.del_points(points, data_time, time)
            vector = time * 1.0
            axis = 0
        else:
            data_res, vec_res = Preprocessing.del_points(points, data_wave, wave)
            axis = 1
            vector = wave * 1.0
        self.assertEqual(data_res.shape[axis], len(vec_res))
        self.assertEqual(data_res.shape[axis], len(vector)-len(points))
    
    @parameterized.expand([[[0, 1, 25, 75], 1],
                           [[0, 1, 25], 0]])
    def test_delPoints_same_size(self, points, axis):
        """
        test in case the number of columns and rows are the same
        """
        datas = np.ones((75, 75))
        data_res, vec_res = Preprocessing.del_points(points, datas, time, axis)
        self.assertEqual(data_res.shape[axis], len(vec_res))
        self.assertEqual(data_res.shape[axis], len(time)-len(points))
    
    @parameterized.expand([[360, None, None],
                           [None, 360, None],
                           [360, 370, 'cut'],
                           [360, 370, 'select']])
    def test_cutColumns(self, mini, maxi, innerdata):
        """
        minimum value lower than 360 idex == 27 (considering cefro index 28 points)
        minimum value lower than 370 idex == 57 (considering cefro index 58 points)
        """
        data_res, vector_res = Preprocessing.cut_columns(data_wave, wave, mini,
                                                         maxi, innerdata)
        if maxi is None:
            self.assertTrue(wave[20] not in vector_res)
            self.assertTrue(vector_res[0] > mini)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 150-28)
            
        if mini is None:
            self.assertTrue(wave[80] not in vector_res)
            self.assertTrue(vector_res[0] < maxi)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 28)
            
        if innerdata == 'cut':
            self.assertTrue(wave[40] not in vector_res)
            self.assertTrue(vector_res[27] < mini)
            self.assertTrue(vector_res[28] > maxi)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 120)
            
        if innerdata == 'select':
            self.assertTrue(wave[20] not in vector_res)
            self.assertTrue(wave[40] in vector_res)
            self.assertTrue(wave[80] not in vector_res)
            self.assertTrue(vector_res[0] > mini)
            self.assertTrue(vector_res[-1] < maxi)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 30)
    
    @parameterized.expand([[None, None, None],
                           [360, 370, None],
                           [None, 370, 'cut'],
                           [360, None, 'select']])
    def test_cutColumns_exception(self, mini, maxi, innerdata):
        """
        testing exceptions are raised
        """
        with self.assertRaises(ExperimentException) as context:
            Preprocessing.cut_columns(data_wave, wave, mini, maxi, innerdata)
            print(context.msg)
    
    @parameterized.expand([[25, None],
                           [None, 40],
                           [25, 40]])
    def test_cutRows(self, mini, maxi):
        """
        minimum value lower than 25 idex == 37 (considering cefro index 38 points)
        minimum value lower than 40 idex == 60 (considering cefro index 61 points)
        """
        data_res, vector_res = Preprocessing.cut_rows(data_time, time, mini, maxi)
        
        if maxi is None and mini is not None:
            self.assertTrue(time[25] not in vector_res)
            self.assertTrue(vector_res[0] > mini)
            self.assertEqual(data_res.shape[0], len(vector_res))
            self.assertEqual(data_res.shape[0], 75-38)
            
        elif mini is None and maxi is not None:
            self.assertTrue(time[65] not in vector_res)
            self.assertTrue(vector_res[0] < maxi)
            self.assertEqual(data_res.shape[0], len(vector_res))
            self.assertEqual(data_res.shape[0], 61)
        else:
            self.assertTrue(time[25] not in vector_res)
            self.assertTrue(time[65] not in vector_res)
            self.assertTrue(vector_res[0] > mini)
            self.assertTrue(vector_res[-1] < maxi)
            self.assertEqual(data_res.shape[0], len(vector_res))
            self.assertEqual(data_res.shape[0], 75-38-14)
    
    @parameterized.expand([[25, 5, 'constant', 5],
                           [14, 10, 'log', 5],
                           [5, 3, 'log', 4],
                           [5, 10, 'constant', 5]])
    def test_averageTimePoints(self, starting_point, step, method, grid_dense):
        time_2 = np.linspace(0, 74, 75)
        data_res, vector_res = Preprocessing.average_time_points(
            data_time, time_2, starting_point, step, method, grid_dense)  
        self.assertTrue(vector_res[starting_point] == starting_point)
        if method == 'cosntant':
            self.assertTrue(vector_res[starting_point+1] == starting_point +
                            step/2)
            self.assertTrue(vector_res[starting_point+2] == starting_point +
                            step/2 + step)
        if method == 'log':
            self.assertTrue(vector_res[starting_point+1] == starting_point+1)
            dif1 = vector_res[starting_point+5] - vector_res[starting_point+4]
            dif2 = vector_res[starting_point+6] - vector_res[starting_point+5]
            self.assertTrue(dif1 < dif2)
    
    @parameterized.expand([[25],
                           [14],
                           [5.3]])
    def shitTime(self, value):
        vector_res = Preprocessing.average_time_points(time, value)
        self.assertTrue(vector_res[0] == time[0] - value)


if __name__ == '__main__':
    unittest.main()
