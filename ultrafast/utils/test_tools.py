import unittest
import numpy as np


class ArrayTestCase(unittest.TestCase):
    def assertNearlyEqualArray(self, array1, array2, decimal, msg=None):
        """
        Asserts the equality of two arrays within decimals
        """
        array1 = np.array(array1)
        array2 = np.array(array2)

        dif = np.array(array1) - np.array(array2)
        expr = (dif < 10 ** (- decimal)).all()

        if not expr:
            msg = self._formatMessage(
                msg,
                'arrays are not neary equal, '
                ' try changing the number of decimals, here it is set '
                f'to {decimal}')

            raise self.failureException(msg)
    
    def assertEqualArray(self, array1, array2, msg=None):
        """
        Asserts the rigorous equality of two arrays
        """
        array1 = np.array(array1)
        array2 = np.array(array2)

        expr = (array1 == array2).all()

        if not expr:
            msg = self._formatMessage(msg, 'arrays are not equal')

            raise self.failureException(msg)
