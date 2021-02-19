import io
import unittest.mock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Tick 
from parameterized import parameterized
from chempyspec.ultrafast.utils.utils import *


data_wave = np.ones((75, 150))
for i in range(data_wave.shape[0]):
    data_wave[i, :] = data_wave[i, :] * i

data_time = np.ones((75, 150))
for i in range(data_time.shape[1]):
    data_time[:, i] = data_time[:, i] * i
wave = np.linspace(351, 350 + 50, 150)
time = np.linspace(0, 49, 75)


def assertEqualArray(array1, array2):
    """
    returns "True" if all elements of two arrays are identical
    """
    if type(array1) == list:
        array1 = np.array(array1)
    if type(array2) == list:
        array2 = np.array(array2)
    value = (array1 == array2).all()
    return value


def assertNearlyEqualArray(array1, array2, decimal):
    """
    returns "True" if all elements of two arrays
    are identical until decimal
    """
    if type(array1) == list:
        array1 = np.array(array1)
    if type(array2) == list:
        array2 = np.array(array2)
    dif = np.array(array1) - np.array(array2)
    value = (dif < 10**(-decimal)).all()
    return value


class TestOutilsFunctions(unittest.TestCase):

    @parameterized.expand([[10, 0, None],
                           [10, 5, None],
                           [10, 5, [370, 380]],
                           [10, 5, [[370, 380], [400, 410]]],
                           [[355, 365, 380], 5, None],
                           [[355, 365, 380], 4, [360, 370]],
                           [[355, 365, 380, 405], 4, [[360, 370], [400, 410]]],
                           ['auto', 5, [380, 390]],
                           ['auto', 5, None],
                           ['auto', 0, None]])
    def test_select_traces(self, space, points, avoid_regions):
        wave2 = np.linspace(351, 351 + 149, 150)
        data_res, vec_res = select_traces(data_time, wave2, space, points, avoid_regions)
        mul = 1
        if avoid_regions is None:
            num = 0 
        else:
            if type(avoid_regions[0]) == list:
                mul = len(avoid_regions)
            num = 1 * mul
        if type(space) == int:
            self.assertEqual(space, vec_res[3] - vec_res[2])
            self.assertEqual(points, vec_res[0] - wave[0])
            self.assertEqual(149 // space + 1 - num, len(vec_res))
        elif type(space) == list:
            self.assertEqual(len(space), data_res.shape[1] + num)
            self.assertTrue(space, vec_res)
        if space == 'auto':
            self.assertEqual(10, len(vec_res))
            self.assertEqual(14, data_res[0][0])
        else:
            self.assertEqual(data_res.shape[1], len(vec_res))
            if type(space) == list:
                val = 4.5 if space[0]-points < wave2[0] else points
            else: 
                val = points
            self.assertEqual(val, data_res[0][0])
        self.assertEqual(len(vec_res), data_res.shape[1])

    @parameterized.expand([[[5, 30], 'constant', 5],
                           [[5, 30], 'constant', 8],
                           [[5, 30], 'exponential', 5],
                           [[5, 30], 'r_exponential', 5],
                           [[5, 30], 'mix_exp', 5]])
    def test_define_weights(self, rango, typo, val):
        """
        index of time > 5: 8
        index of time > 10: 16
        index of time < 30: 45
        """
        dicto = define_weights(time, rango, typo, val)
        vec_res = dicto['vector']
        self.assertEqual(dicto['apply'], True)
        self.assertEqual(len(dicto), 5)
        self.assertEqual(len(vec_res), len(time))
        index = 8 if typo != 'r_exponential' else 45
        self.assertEqual(vec_res[index], val)
        self.assertEqual(vec_res[7], 1)
        self.assertEqual(vec_res[46], 1)


name = 'Lab book test'
book_notes = LabBook(name=name)


@book_annotate(book_notes, True)
def fun_test1(val):
    return val * 1.0


@book_annotate(book_notes, True)
def fun_test2(val1, val2):
    return val1 * val2


@book_annotate(book_notes, False)
def fun_test3(val1, val2):
    return val1 + val2


def has_attribute(objecto, key):
    if hasattr(objecto, key):
        return True
    else:
        return False


def attribute_equal(objecto, key, val):
    if hasattr(objecto, key) and getattr(objecto, key) == val:
        return True
    else:
        return False


class TestBookAnnotate(unittest.TestCase):
    """
    testing how ro annotate with book_annotate decorator to a LabBook instance
    """
    def test_book_annotate(self):
        fun_test1(5)
        self.assertTrue(has_attribute(book_notes, 'fun_test1'))
        self.assertTrue(attribute_equal(book_notes, 'fun_test1', 'val = 5'))
        fun_test1(4)
        self.assertTrue(attribute_equal(book_notes, 'fun_test1', ['val = 5', 'val = 4']))
        fun_test2(5, 4)
        self.assertTrue(has_attribute(book_notes, 'fun_test2'))
        self.assertTrue(attribute_equal(book_notes, 'fun_test2', 'val1 = 5, val2 = 4'))
        fun_test2(4, 5)
        self.assertTrue(attribute_equal(book_notes, 'fun_test2', ['val1 = 5, val2 = 4', 'val1 = 4, val2 = 5']))
        fun_test3(5, 4)
        self.assertTrue(has_attribute(book_notes, 'fun_test3'))
        self.assertTrue(attribute_equal(book_notes, 'fun_test3', 'val1 = 5, val2 = 4'))
        fun_test3(5, 4)
        self.assertTrue(attribute_equal(book_notes, 'fun_test3', 'val1 = 5, val2 = 4'))


book = LabBook(name=name)
book.test = 5
book2 = LabBook(name=name, test=[5, 5, 4], valu=5)
book3 = LabBook(name=name, test=[5, 5, 4], valu=5)
print_valu = '\t valu:\n\t\t 5\n\n'
print_test = '\t test:\n\t\t 5\n\t\t 5\n\t\t 4\n\n'
print_book3_cr = f'{name}\n-------------\n{print_test}{print_valu}\t creation:\n\t\t {book3.creation}\n\n'
print_book3 = f'{name}\n-------------\n{print_test}{print_valu}'


class TestLabBook(unittest.TestCase):
    """
    testing LabBook
    """

    def test___init__(self):
        self.assertEqual(book.name, name)

    @parameterized.expand([['test', 5, True, [5, 5]],
                           ['valu', 5, False, 5],
                           ['test', 4, True, [5, 5, 4]],
                           ['valu', 5, False, 5]])
    def test_set_attribute(self, key, val, extend, result):
        book.__setattr__(key, val, extend)
        if type(result) == list:
            res = assertEqualArray(getattr(book, key), result)
            self.assertTrue(res, True)
        else:
            self.assertEqual(getattr(book, key), result)

    @parameterized.expand([['test', 1, [5, 4]],
                           ['valu', 'all', False],
                           ['test', 0, [4]]])
    def test_delete(self, key, val, result):
        book2.delete(key, val)
        if type(val) == int:
            res = assertEqualArray(getattr(book2, key), result)
            self.assertTrue(res, True)
        else:
            self.assertEqual(hasattr(book2, key), result)
    
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, key, expected_output, mock_stdout):
        if key is True:
            book3.print()
        elif key is False:
            book3.print(False)
        else:
            book3._print_attribute(key)
        self.assertEqual(mock_stdout.getvalue(), expected_output)
        
    @parameterized.expand([['valu', print_valu],
                           ['test', print_test]])
    def test__print_attribute(self, key, result):
        self.assert_stdout(key, result)
    
    @parameterized.expand([[True, print_book3_cr],
                           [False, print_book3]])     
    def test_print_attribute(self, val, result):
        self.assert_stdout(val, result)


container = UnvariableContainer()
container.already = 'already'
container.cannot = 'cannot'


class TestUnvariableContainer(unittest.TestCase):
    """
    testing UnvariableContainer and froze_it
    """

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, key, new, expected_output, mock_stdout):
        container.__setattr__(key, new)
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    @parameterized.expand([['new'],
                           ['old'],
                           ['hola']])
    def test_set_new_attribute(self, val):
        container.__setattr__(val, val)
        self.assertTrue(has_attribute(container, val))
        self.assertTrue(attribute_equal(container, val, val))

    @parameterized.expand([['already', 'not already'],
                           ['cannot', 'not cannot']])
    def test_reset_attribute(self, val, new):
        stdout = f'Class UnvariableContainer is frozen. Cannot modified {val} = {new}\n'
        self.assert_stdout(val, new, stdout)
        self.assertTrue(has_attribute(container, val))
        self.assertTrue(attribute_equal(container, val, val))


data_fig = np.ones((5, 50))
x = np.linspace(20, 20+50, 50)
for i in range(data_fig.shape[0]):
    data_fig[i, :] = 2.5*np.sin(x)/(i+1)


def paint_figure():
    fig, ax = plt.subplots(1, 1)
    for step in range(data_fig.shape[0]):
        ax.plot(x, data_fig[step, :])
    return fig, ax


class TestFiguresFormating(unittest.TestCase):

    @parameterized.expand([['x label', 'y label'],
                           ['intensity', 'sin x'],
                           [None, 'sin x'],
                           ['intensity', None]])
    def test__axisLabels(self, x_label, y_label):
        fig, ax = paint_figure()
        FiguresFormating.axis_labels(ax, x_label, y_label)
        if x_label is None:
            x_label = 'X vector'
        if y_label is None:
            y_label = 'Y vector'
        x_l = ax.xaxis.get_label().get_text()
        y_l = ax.yaxis.get_label().get_text()
        self.assertEqual(x_l, x_label)
        self.assertEqual(y_l, y_label)
        plt.close(fig)

    def test__coverExcitation(self):
        fig, ax = paint_figure()
        FiguresFormating.cover_excitation(ax, [30, 40], x)
        patch = ax.patches[0]
        self.assertTrue(patch.fill)
        self.assertEqual(patch.xy[0], 29.20408163265306)
        self.assertEqual(patch.xy[1], -2.610639568376848)
        self.assertEqual(patch.zorder, np.inf)
        plt.close(fig)

    @parameterized.expand([[True, True, 50],
                           [False, True, 50],
                           [False, False, 50],
                           [True, False, 25],
                           [True, True, 25]])
    def test__formatFigure(self, x_tight, set_ylim, val):
        fig, ax = paint_figure()
        FiguresFormating.format_figure(ax, data_fig, x, size=14, x_tight=x_tight, set_ylim=set_ylim, val=val)
        lines = ax.lines
        prop = Tick.properties(ax)
        self.assertEqual(len(lines), 6)
        self.assertTrue(len(prop['yminorticklabels']) != 0)
        if x_tight:
            self.assertTrue(assertEqualArray(prop['xlim'], [x[0], x[-1]]))  
        else:
            self.assertTrue(assertEqualArray(prop['xlim'], [x[0] - x[-1]/val, x[-1] + x[-1]/val]))  
        if set_ylim:
            vec = [np.min(data_fig) - abs(np.min(data_fig) * 0.1), np.max(data_fig) + np.max(data_fig) * 0.1]
            self.assertTrue(assertEqualArray(prop['ylim'], vec)) 
        plt.close(fig) 


wave_test = DataSetCreator.generate_wavelength(50, 350, 100)
shape_test = DataSetCreator.generate_shape(5, wave_test, signe=0, sigma=50)
time_test = DataSetCreator.generate_time(-1, 1000, 120, 'lin-log')
array_test = DataSetCreator.generate_dataset(shape_test, time_test, 0.12)


class TestDataSetCreator(unittest.TestCase):

    @parameterized.expand([[-10, 100, 50],
                           [10, -100, 50],
                           [-10, -100, 50]])
    def test_generate_wavelength(self, init, final, point):
        array = DataSetCreator.generate_wavelength(init, final, point)
        self.assertEqual(abs(init), array[0])
        self.assertEqual(abs(final), array[-1])
        self.assertEqual(len(array), point)

    @parameterized.expand([[-10, 100, 50, 'log'],
                           [10, 100, 50, 'linear'],
                           [-10, 100, 50, 'lin-log'],
                           [-10, 100, 95, 'lin-log'],
                           [-10, 100, 77, 'lin-log'],
                           [-10, 100, 120, 'lin-log']])
    def test_generate_time(self, init, final, point, space):
        array = DataSetCreator.generate_time(init, final, point, space)
        self.assertEqual(init, array[0])
        self.assertEqual(final, int(array[-1]))
        self.assertEqual(len(array), point)
        if space == 'lin-log' and init < 1:
            self.assertEqual(round(abs(array[0]-array[1]), 2), round(abs(array[1]-array[2]), 2))

    @parameterized.expand([[[1, 10, 50], -1],
                           [[1, 10], 1],
                           [[1, 10, 50, 150], 0]])
    def test_generate_shape(self, taus, signe):
        array = DataSetCreator.generate_shape(10, wave_test, scale=100, taus=taus, signe=signe, sigma=2.25)
        self.assertTrue(assertEqualArray(array.index, taus))
        self.assertEqual(array.shape[0], len(taus))
        self.assertEqual(array.shape[1], len(wave_test))
        if signe == 1:
            self.assertTrue(array.min().min() >= 0)
        if signe == -1:
            self.assertTrue(array.max().max() <= 0.001)
        if signe == 0:
            self.assertTrue(array.max().max() >= 0)
            self.assertTrue(array.min().min() <= 0)

    def test_generate_dataset(self):
        self.assertEqual(array_test.shape[0], len(time_test))
        self.assertEqual(array_test.shape[1], len(wave_test))
        self.assertTrue(assertNearlyEqualArray([float(i) for i in array_test.index], time_test, 4))
        self.assertTrue(assertNearlyEqualArray([float(i) for i in array_test.columns], wave_test, 1))


if __name__ == '__main__':
    unittest.main()
