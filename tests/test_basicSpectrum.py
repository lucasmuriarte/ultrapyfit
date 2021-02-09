import unittest
import numpy as np
from ultrafast.BasicSpectrumClass import BasicSpectrum
import sys

x = np.linspace(250, 800, 800 - 249)


def gauss(x, amp, cen, sigma):
    "basic gaussian"
    return amp * np.exp(-(x - cen)**2 / (2. * sigma**2))


np.random.RandomState(111)
y = gauss(x, 1, 500, 50) + gauss(x, 1.2, 550, 20) + gauss(x, 0.6,
                                                          630, 30) - 0.2 + np.random.normal(size=len(x), scale=0.01)
y2 = gauss(x, 1.1, 400, 40) + gauss(x, 1, 550, 60) + gauss(x, 0.6,
                                                           580, 40) + 0.2 + np.random.normal(size=len(x), scale=0.01)
spec = BasicSpectrum(x, y)
spec2 = BasicSpectrum(x, y2)


class TestBasicSpectrum(unittest.TestCase):

    def assertEqualArray(self, array1, array2):
        ''' returns "True" if all elements of two arrays are identical'''
        value = (array1 == array2).all()
        return value

    def test__minMaxIndex(self):
        low, high = spec._minMaxIndex(500, 700)
        self.assertEqual(low, 250)
        self.assertEqual(high, 451)

    def test_cut_low(self):
        res = spec.cut(300, itself=False)
        self.assertEqual(res.data_table['x'].values[0], 300)
        self.assertEqual(res.data_table['x'].values[-1], x[-1])

    def test_cut_high(self):
        res = spec.cut(high=300, itself=False)
        self.assertEqual(res.data_table['x'].values[0], x[0])
        self.assertEqual(res.data_table['x'].values[-1], 300)

    def test_cut_low_high(self):
        res = spec.cut(300, 500, itself=False)
        self.assertEqual(res.data_table['x'].values[0], 300)
        self.assertEqual(res.data_table['x'].values[-1], 500)

    def test_minus(self):
        resta = spec - spec2
        self.assertTrue(
            self.assertEqualArray(
                resta.data_table['y'].values,
                y - y2))

    def test_plus(self):
        suma = spec + spec2
        self.assertTrue(
            self.assertEqualArray(
                suma.data_table['y'].values,
                y + y2))

    def test_mul(self):
        mul = spec * 3
        self.assertTrue(
            self.assertEqualArray(
                mul.data_table['y'].values,
                y * 3))

    def test_div(self):
        div = spec / 5
        self.assertTrue(
            self.assertEqualArray(
                div.data_table['y'].values,
                y / 5))

    def test_obtainValue(self):
        self.assertEqual(round(spec.obtainValue(500), 3), round(y[250], 3))
        self.assertEqual(round(spec.obtainValue(700), 3), round(y[450], 3))

    def test_obtainMax(self):
        self.assertEqual(round(spec.obtainMax()[1], 3), round(y.max(), 3))

    def test_obtainMax_high(self):
        self.assertEqual(round(spec.obtainMax(high=600)[
                         1], 3), round(y[:350].max(), 3))

    def test_obtainMax_low(self):
        self.assertEqual(round(spec.obtainMax(low=600)[
                         1], 3), round(y[350:].max(), 3))

    def test_obtainMin(self):
        self.assertEqual(round(spec.obtainMin()[1], 3), round(y.min(), 3))

    def test_obtainMin_high(self):
        self.assertEqual(round(spec.obtainMin(high=600)[
                         1], 3), round(y[:350].min(), 3))

    def test_obtainMin_low(self):
        self.assertEqual(round(spec.obtainMin(low=600)[
                         1], 3), round(y[350:].min(), 3))

    def test_focePositive(self):
        positive = spec.forcePositive(itself=False)
        self.assertTrue((positive.data_table['y'] >= 0).all())

    def test_normalize(self):
        norm = spec.normalize(itself=False)
        value = norm.obtainMax()
        self.assertEqual(norm.obtainValue(value[0]), 1)

    def test_normalize_value(self):
        norm = spec.normalize(500, itself=False)
        self.assertEqual(norm.obtainValue(500), 1)

    def test_normalize_value_low_high(self):
        norm = spec.normalize(low=600, high=750, itself=False)
        value = norm.obtainMax(low=600)
        self.assertEqual(norm.obtainValue(value[0]), 1)

    def test_calculateArea(self):
        self.assertTrue(
            self.assertEqualArray(
                spec.calculateArea(),
                np.trapz(
                    y,
                    x)))

    def test_calculateArea_low_high(self):
        self.assertTrue(spec.calculateArea(500, 700),
                        np.trapz(y[250:451], x[250:451]))

    def test_averageOfRange_low_high(self):
        self.assertEqual(round(spec.averageOfRange(500, 700),
                               5), round(np.mean(y[250:451]), 5))

    def test_averageOfRange(self):
        self.assertEqual(round(spec.averageOfRange(), 5), round(np.mean(y), 5))

    def test_baselineCorrection(self):
        norm = spec.baselineCorrection(730, itself=False)
        self.assertTrue(self.assertEqualArray(
            norm.data_table['y'], y - np.mean(y[480:])))


if __name__ == '__main__':
    unittest.main()
