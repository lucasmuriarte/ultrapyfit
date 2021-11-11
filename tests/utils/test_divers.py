import io
import unittest.mock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Tick
from parameterized import parameterized
from ultrafast.utils.divers import *
from ultrafast.utils.test_tools import ArrayTestCase


class TestOutilsFunctions(ArrayTestCase):
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
        [10, 0, None],
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
        data_res, vec_res = select_traces(
            self.data_time, wave2, space, points, avoid_regions)

        mul = 1

        if avoid_regions is None:
            num = 0

        else:
            if type(avoid_regions[0]) == list:
                mul = len(avoid_regions)

            num = 1 * mul

        if type(space) == int:
            self.assertEqual(space, vec_res[3] - vec_res[2])
            self.assertEqual(points, vec_res[0] - self.wave[0])
            self.assertEqual(149 // space + 1 - num, len(vec_res))

        elif type(space) == list:
            self.assertEqual(len(space), data_res.shape[1] + num)
            self.assertTrue(space, vec_res)  # TODO assertTrue ?

        if space == 'auto':
            self.assertEqual(10, len(vec_res))
            self.assertEqual(14, data_res[0][0])

        else:
            self.assertEqual(data_res.shape[1], len(vec_res))
            if type(space) == list:
                val = 4.5 if space[0] - points < wave2[0] else points

            else:
                val = points

            self.assertEqual(val, data_res[0][0])
            
        self.assertEqual(len(vec_res), data_res.shape[1])

    @parameterized.expand([
        [[5, 30], 'constant', 5],
        [[5, 30], 'constant', 8],
        [[5, 30], 'exponential', 5],
        [[5, 30], 'r_exponential', 5],
        [[5, 30], 'mix_exp', 5]])
    def test_define_weights(self, rango, typo, val):
        # TODO rango
        """
        index of time > 5: 8
        index of time > 10: 16
        index of time < 30: 45
        """
        dicto = define_weights(self.time, rango, typo, val)
        vec_res = dicto['vector']

        self.assertEqual(dicto['apply'], True)
        self.assertEqual(len(dicto), 5)
        self.assertEqual(len(vec_res), len(self.time))

        index = 8 if typo != 'r_exponential' else 45

        self.assertEqual(vec_res[index], val)
        self.assertEqual(vec_res[7], 1)
        self.assertEqual(vec_res[46], 1)


class TestLabBook(ArrayTestCase):
    """
    testing how ro annotate with book_annotate decorator to a LabBook instance
    """
    name = 'Lab book test'
    book_notes = LabBook(name=name)

    @classmethod
    @book_annotate(book_notes, True)
    def fun_test1(cls, val):
        return val * 1.0


    @classmethod
    @book_annotate(book_notes, True)
    def fun_test2(cls, val1, val2):
        return val1 * val2


    @classmethod
    @book_annotate(book_notes, False)
    def fun_test3(cls, val1, val2):
        return val1 + val2


    @classmethod
    def has_attribute(cls, objecto, key):
        if hasattr(objecto, key):
            return True
        else:
            return False

    @classmethod
    def attribute_equal(cls, objecto, key, val):
        if hasattr(objecto, key) and getattr(objecto, key) == val:
            return True
        else:
            return False

    def test_book_annotate(self):
        self.fun_test1(5)
        self.assertTrue(
            self.has_attribute(self.book_notes, 'fun_test1')
        )

        self.assertTrue(
            self.attribute_equal(self.book_notes, 'fun_test1', 'val = 5')
        )

        self.fun_test1(4)
        self.assertTrue(
            self.attribute_equal(
                self.book_notes, 'fun_test1', ['val = 5', 'val = 4'])
        )

        self.fun_test2(5, 4)
        self.assertTrue(
            self.has_attribute(self.book_notes, 'fun_test2')
        )

        self.assertTrue(
            self.attribute_equal(
                self.book_notes, 'fun_test2', 'val1 = 5, val2 = 4')
        )

        self.fun_test2(4, 5)
        self.assertTrue(
            self.attribute_equal(
                self.book_notes, 'fun_test2',
                ['val1 = 5, val2 = 4', 'val1 = 4, val2 = 5'])
        )

        self.fun_test3(5, 4)
        self.assertTrue(
            self.has_attribute(self.book_notes, 'fun_test3')
        )

        self.assertTrue(
            self.attribute_equal(
                self.book_notes, 'fun_test3', 'val1 = 5, val2 = 4')
        )

        self.fun_test3(5, 4)
        self.assertTrue(
            self.attribute_equal(
                self.book_notes, 'fun_test3', 'val1 = 5, val2 = 4')
        )

    """
    testing LabBook
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.book = LabBook(name=cls.name)
        cls.book.test = 5
        cls.book2 = LabBook(name=cls.name, test=[5, 5, 4], value=5)
        cls.book3 = LabBook(name=cls.name, test=[5, 5, 4], value=5)

        cls.print_value = '\t value:\n\t\t 5\n\n'

        cls.print_test = '\t test:\n\t\t 5\n\t\t 5\n\t\t 4\n\n'

        cls.print_book3_cr = f'{cls.name}\n'
        cls.print_book3_cr += f'-------------\n'
        cls.print_book3_cr += f'{cls.print_test}{cls.print_value}\t'
        cls.print_book3_cr += f' creation:\n\t\t {cls.book3.creation}\n\n'

        cls.print_book3 = \
            f'{cls.name}\n-------------\n{cls.print_test}{cls.print_value}'

    def test___init__(self):
        self.assertEqual(self.book.name, self.name)

    @parameterized.expand([
        ['test', 5, True, [5, 5]],
        ['value', 5, False, 5],
        ['test', 4, True, [5, 5, 4]],
        ['value', 5, False, 5]])
    def test_set_attribute(self, key, val, extend, result):
        self.book.__setattr__(key, val, extend)

        if type(result) == list:
            self.assertEqualArray(getattr(self.book, key), result)

        else:
            self.assertEqual(getattr(self.book, key), result)

    @parameterized.expand([
        ['test', 1, [5, 4]],
        ['value', 'all', False],
        ['test', 0, [4]]])
    def test_delete(self, key, val, result):
        self.book2.delete(key, val)
        if type(val) == int:
            self.assertEqualArray(getattr(self.book2, key), result)
        else:
            self.assertEqual(hasattr(self.book2, key), result)
    
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, key, expected_output, mock_stdout):
        if key is True:
            self.book3.print()
        elif key is False:
            self.book3.print(False)
        else:
            self.book3._print_attribute(key)
        self.assertEqual(mock_stdout.getvalue(), expected_output)
        
    @parameterized.expand([
        ['value', 'print_value'],
        ['test', 'print_test']])
    def test__print_attribute(self, key, result):
        if result == 'print_value':
            result = self.print_value

        elif result == 'print_test':
            result = self.print_test

        self.assert_stdout(key, result)
    
    @parameterized.expand([[True, 'print_book3_cr'], [False, 'print_book3']])
    def test_print_attribute(self, val, result):
        if result == 'print_book3_cr':
            result = self.print_book3_cr

        elif result == 'print_book3':
            result = self.print_book3

        self.assert_stdout(val, result)


class TestUnvariableContainer(unittest.TestCase):
    """
    testing UnvariableContainer and froze_it
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.container = UnvariableContainer()
        cls.container.already = 'already'
        cls.container.cannot = 'cannot'


    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, key, new, expected_output, mock_stdout):
        self.container.__setattr__(key, new)
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    @parameterized.expand([['new'], ['old'], ['hola']])
    def test_set_new_attribute(self, val):
        self.container.__setattr__(val, val)
        self.assertTrue(has_attribute(self.container, val))
        self.assertTrue(attribute_equal(self.container, val, val))

    @parameterized.expand([
        ['already', 'not already'],
        ['cannot', 'not cannot']])
    def test_reset_attribute(self, val, new):
        stdout = f'Class UnvariableContainer is frozen.'
        ' Cannot modified {val} = {new}\n'

        self.assert_stdout(val, new, stdout)
        self.assertTrue(has_attribute(self.container, val))
        self.assertTrue(attribute_equal(self.container, val, val))


class TestFiguresFormating(ArrayTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_fig = np.ones((5, 50))
        cls.x = np.linspace(20, 20 + 50, 50)

        for i in range(cls.data_fig.shape[0]):
            cls.data_fig[i, :] = 2.5 * np.sin(cls.x) / (i + 1)

    @classmethod
    def paint_figure(cls):
        fig, ax = plt.subplots(1, 1)

        for step in range(cls.data_fig.shape[0]):
            ax.plot(cls.x, cls.data_fig[step, :])

        return fig, ax

    @parameterized.expand([['x label', 'y label'],
                           ['intensity', 'sin x'],
                           [None, 'sin x'],
                           ['intensity', None]])
    def test__axisLabels(self, x_label, y_label):
        fig, ax = self.paint_figure()

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
        fig, ax = self.paint_figure()

        FiguresFormating.cover_excitation(ax, [30, 40], self.x)

        patch = ax.patches[0]

        self.assertTrue(patch.fill)
        self.assertEqual(patch.xy[0], 29.20408163265306)
        self.assertEqual(patch.xy[1], -2.610639568376848)
        self.assertEqual(patch.zorder, np.inf)

        plt.close(fig)

    @parameterized.expand([
        [True, True, 50],
        [False, True, 50],
        [False, False, 50],
        [True, False, 25],
        [True, True, 25]])
    def test__formatFigure(self, x_tight, set_ylim, val):
        fig, ax = self.paint_figure()

        FiguresFormating.format_figure(
            ax,
            self.data_fig,
            self.x,
            size=14,
            x_tight=x_tight,
            set_ylim=set_ylim,
            val=val)

        lines = ax.lines
        prop = Tick.properties(ax)

        self.assertEqual(len(lines), 6)
        self.assertNotEqual(len(prop['yminorticklabels']), 0)

        if x_tight:
            self.assertEqualArray(prop['xlim'], [self.x[0], self.x[-1]])

        else:
            self.assertEqualArray(
                prop['xlim'],
                [
                    self.x[0] - self.x[-1] / val,
                    self.x[-1] + self.x[-1] / val
                ])

        if set_ylim:
            vec = [
                np.min(self.data_fig) - abs(np.min(self.data_fig) * 0.1),
                np.max(self.data_fig) + np.max(self.data_fig) * 0.1
            ]

            self.assertEqualArray(prop['ylim'], vec)

        plt.close(fig)


class TestDataSetCreator(ArrayTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wave_test = \
            DataSetCreator.generate_wavelength(
                50, 350, 100)

        cls.shape_test = \
            DataSetCreator.generate_shape(
                5, cls.wave_test, signe=0, sigma=50)

        cls.time_test = \
            DataSetCreator.generate_time(
                -1, 1000, 120, 'lin-log')

        cls.array_test = \
            DataSetCreator.generate_dataset(
                cls.shape_test, cls.time_test, 0.12)

    @parameterized.expand([
        [-10, 100, 50],
        [10, -100, 50],
        [-10, -100, 50]])
    def test_generate_wavelength(self, init, final, point):
        array = DataSetCreator.generate_wavelength(init, final, point)

        self.assertEqual(abs(init), array[0])
        self.assertEqual(abs(final), array[-1])
        self.assertEqual(len(array), point)

    @parameterized.expand([
        [-10, 100, 50, 'log'],
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
            self.assertEqual(
                round(abs(array[0] - array[1]), 2),
                round(abs(array[1] - array[2]), 2)
            )

    @parameterized.expand([[[1, 10, 50], -1],
                           [[1, 10], 1],
                           [[1, 10, 50, 150], 0]])
    def test_generate_shape(self, taus, signe):
        array = DataSetCreator.generate_shape(
            10,
            self.wave_test,
            taus,
            scale=100,
            signe=signe,
            sigma=2.25
        )

        self.assertEqualArray(array.index, taus)
        self.assertEqual(array.shape[0], len(taus))
        self.assertEqual(array.shape[1], len(self.wave_test))

        if signe == 1:
            self.assertGreaterEqual(array.min().min(), 0)

        if signe == -1:
            self.assertLessEqual(array.max().max(), 0.001)

        if signe == 0:
            self.assertGreaterEqual(array.max().max(), 0)
            self.assertLessEqual(array.min().min(), 0)

    def test_generate_dataset(self):
        self.assertEqual(
            self.array_test.shape[0], len(self.time_test))

        self.assertEqual(
            self.array_test.shape[1], len(self.wave_test))

        self.assertNearlyEqualArray(
            [float(i) for i in self.array_test.index],
            self.time_test, 4)

        self.assertNearlyEqualArray(
            [float(i) for i in self.array_test.columns],
            self.wave_test, 1)


if __name__ == '__main__':
    unittest.main()
