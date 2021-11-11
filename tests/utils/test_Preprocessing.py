import unittest
import numpy as np
from parameterized import parameterized
import sys

from ultrafast.utils.test_tools import ArrayTestCase
from ultrafast.utils.Preprocessing import Preprocessing, ExperimentException


class TestPreprocessing(ArrayTestCase):
    """
    test for Preprocessing Class
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_wave = np.ones((75, 150))

        for i in range(cls.data_wave.shape[0]):
            cls.data_wave[i, :] = cls.data_wave[i, :] * i
            
        cls.data_time = np.ones((75, 150))

        for i in range(cls.data_time.shape[1]):
            cls.data_time[:, i] = cls.data_time[:, i] * i

        cls.wave = np.linspace(351, 350 + 50, 150)
        cls.time = np.linspace(0, 49, 75)
        
    @parameterized.expand([
        [0, True],
        [0, False],
        [[3, 5], False],
        [[2, 5], False],
        [3, False],
        [5, False],
        [2, False],
        [1, True]])
    def test_baselineSubstraction(self, number_spec, only_one):
        # TODO resta
        """
        test for baseline sbtarction considering all rows of data_wave have
        only one number increassing from 0 to 74
        """
        result = Preprocessing.baseline_substraction(
            self.data_wave, number_spec, only_one)
        exp = self.data_wave * 0.0

        if only_one or number_spec == 0:
            resta = np.ones((1, 150)) * number_spec

        else:
            if type(number_spec) == int:
                number_spec = [0, number_spec - 1]

            resta = np.ones((1, 150)) * np.mean(number_spec)

        for i in range(self.data_wave.shape[0]):
            exp[i, :] = self.data_wave[i, :] - resta

        self.assertNearlyEqualArray(result, exp, 7)
    
    @parameterized.expand([
        [[0, 1, 25, 75]],
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
            data_res, vec_res = Preprocessing.del_points(
                points, self.data_time, self.time)

            vector = self.time * 1.0
            axis = 0

        else:
            data_res, vec_res = Preprocessing.del_points(
                points, self.data_wave, self.wave)

            axis = 1
            vector = self.wave * 1.0

        self.assertEqual(data_res.shape[axis], len(vec_res))
        self.assertEqual(data_res.shape[axis], len(vector) - len(points))
    
    @parameterized.expand([
        [[0, 1, 25, 75], 1],
        [[0, 1, 25], 0]])
    def test_delPoints_same_size(self, points, axis):
        """
        test in case the number of columns and rows are the same
        """
        datas = np.ones((75, 75))
        data_res, vec_res = Preprocessing.del_points(
            points, datas, self.time, axis)

        self.assertEqual(data_res.shape[axis], len(vec_res))
        self.assertEqual(data_res.shape[axis], len(self.time) - len(points))
    
    @parameterized.expand([
        [360, None, None],
        [None, 360, None],
        [360, 370, 'cut'],
        [360, 370, 'select']])
    def test_cutColumns(self, mini, maxi, innerdata):
        """
        minimum value lower than 360 idex == 27 (considering cefro index 28
        points)
        minimum value lower than 370 idex == 57 (considering cefro index 58
        points)
        """
        data_res, vector_res = Preprocessing.cut_columns(
            self.data_wave, self.wave, mini, maxi, innerdata)
            
        if maxi is None:
            self.assertNotIn(self.wave[20], vector_res)
            self.assertGreater(vector_res[0], mini)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 150 - 28)
            
        if mini is None:
            self.assertNotIn(self.wave[80], vector_res)
            self.assertLess(vector_res[0], maxi)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 28)
            
        if innerdata == 'cut':
            self.assertNotIn(self.wave[40], vector_res)
            self.assertLess(vector_res[27], mini)
            self.assertGreater(vector_res[28], maxi)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 120)
            
        if innerdata == 'select':
            self.assertNotIn(self.wave[20], vector_res)
            self.assertIn(self.wave[40], vector_res)
            self.assertNotIn(self.wave[80], vector_res)
            self.assertGreater(vector_res[0], mini)
            self.assertLess(vector_res[-1], maxi)
            self.assertEqual(data_res.shape[1], len(vector_res))
            self.assertEqual(data_res.shape[1], 30)
    
    @parameterized.expand([
        [None, None, None],
        [360, 370, None],
        [None, 370, 'cut'],
        [360, None, 'select']])
    def test_cutColumns_exception(self, mini, maxi, innerdata):
        """
        testing exceptions are raised
        """
        with self.assertRaises(ExperimentException) as context:
            Preprocessing.cut_columns(
                self.data_wave, self.wave, mini, maxi, innerdata)
            print(context.msg)
    
    @parameterized.expand([
        [25, None],
        [None, 40],
        [25, 40]])
    def test_cutRows(self, mini, maxi):
        """
        minimum value lower than 25 idex == 37 (considering cefro index 38
        points)
        minimum value lower than 40 idex == 60 (considering cefro index 61
        points)
        """
        data_res, vector_res = Preprocessing.cut_rows(
            self.data_time, self.time, mini, maxi)
        
        if maxi is None and mini is not None:
            self.assertNotIn(self.time[25], vector_res)
            self.assertGreater(vector_res[0], mini)
            self.assertEqual(data_res.shape[0], len(vector_res))
            self.assertEqual(data_res.shape[0], 75 - 38)
            
        elif mini is None and maxi is not None:
            self.assertNotIn(self.time[65], vector_res)
            self.assertLess(vector_res[0], maxi)
            self.assertEqual(data_res.shape[0], len(vector_res))
            self.assertEqual(data_res.shape[0], 61)
        else:
            self.assertNotIn(self.time[25], vector_res)
            self.assertNotIn(self.time[65], vector_res)
            self.assertGreater(vector_res[0], mini)
            self.assertLess(vector_res[-1], maxi)
            self.assertEqual(data_res.shape[0], len(vector_res))
            self.assertEqual(data_res.shape[0], 75 - 38 - 14)
    
    @parameterized.expand([
        [25, 5, 'constant', 5],
        [14, 10, 'log', 5],
        [5, 3, 'log', 4],
        [5, 10, 'constant', 5]])
    def test_averageTimePoints(self, starting_point, step, method, grid_dense):
        time_2 = np.linspace(0, 74, 75)

        data_res, vector_res = Preprocessing.average_time_points(
            self.data_time, time_2, starting_point, step, method, grid_dense)

        self.assertEqual(vector_res[starting_point], starting_point)

        if method == 'cosntant':
            self.assertEqual(
                vector_res[starting_point + 1],
                starting_point + step / 2)

            self.assertEqual(
                vector_res[starting_point + 2],
                starting_point + step / 2 + step)

        if method == 'log':
            self.assertEqual(
                vector_res[starting_point + 1], starting_point + 1)

            dif1 = vector_res[starting_point + 5] \
                - vector_res[starting_point + 4]

            dif2 = vector_res[starting_point + 6] \
                - vector_res[starting_point + 5]

            self.assertLess(dif1, dif2)
    
    @parameterized.expand([[25],
                           [14],
                           [5.3]])
    def shitTime(self, value):
        vector_res = Preprocessing.average_time_points(self.time, value)
        self.assertEqual(vector_res[0], self.time[0] - value)


if __name__ == '__main__':
    unittest.main()
