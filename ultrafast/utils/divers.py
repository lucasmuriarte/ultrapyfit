# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:13:09 2020

@author: Lucas
"""

from matplotlib.patches import Rectangle
from functools import wraps
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import re
import datetime
from ultrafast.utils.Preprocessing import ExperimentException
from ultrafast.fit.ModelCreator import ModelCreator
from enum import Enum
from decimal import Decimal
import traceback
import pathlib

class TimeMultiplicator(Enum):
    y = 1E-24
    z = 1E-21
    a = 1E-18 # ato
    f = 1E-15 # femto
    p = 1E-12 # pico
    n = 1E-9 # nano
    µ = 1E-6 # micro
    m = 1E-3 # mili
    s = 1E0 # one
    
    @staticmethod
    def chose(value: str):
        try:
            value = value.lower()
            if value[0] == 'u' or 'micro' in value:
                value = 'μ'
            return TimeMultiplicator(TimeMultiplicator._member_map_[value[0]])
        except Exception:
            msg = "the value should start by one of the letters in " \
                  "TimeMultiplicator"
            raise ExperimentException(msg)


class TimeUnitFormater:
    """
    Class to format a time number in a string with the corresponding unit, after
    indicating the unit measure.
    
    
    i.e.:
        formater = TimeUnitFormater("m") # formater for milisecond measurement
        
        formater.value_formated(3046)
        >>> '3.05 s'
        
        formater.value_formated(3046E-7)
        >>> '0.30 μs'
        
        formater.value_formated(3046E3)
        >>> '50.77 mins' 
        
        formater.value_formated(3046E3, 4)
        >>> '50.7667 mins'        
    """
    def __init__(self, multiplicator: TimeMultiplicator = TimeMultiplicator.p):
        if type(multiplicator) == type(TimeMultiplicator):
            self._multiplicator = multiplicator
        else:
            try:
                self.multiplicator = multiplicator
            except Exception:
                msg = "TimeMultiplicator instantiation error. Pass a " \
                      "TimeMultiplicator name as a string or a " \
                      "TimeMultiplicator object"
                raise ExperimentException(msg)
    
    @property    
    def multiplicator(self):
        return self._multiplicator.value
    
    @multiplicator.setter
    def multiplicator(self, name):
        self._multiplicator = TimeMultiplicator.chose(name)

    def value_formated(self, value, decimal=2):
        """
        Function to transform the number to string
        Parameters
        ----------
        value: int or float
            Value to be formatted
            
        decimal: int (default 2)
            number of decimal numbers of the output string
            
        """
        return self._transform(value, decimal)
    
    def _transform(self, value, decimal):
        """
        Real function to transforming the number to string
        """
        negative = False
        if value == 0:
            inter = 0
            name = f'{self._multiplicator.name}s'
            if name == 'ss':
                name = name[0]
        else:
            turns = 0
            if value < 0:
                negative = True
            inter = abs(value)
            if abs(value) > 999.9:
                multi = 1E3
                while inter > 999.9:
                    inter /= 1E3
                    turns += 1
            elif abs(value) < 0.01:
                multi = 1E-3
                while inter < 0.01:
                    inter *= 1E3
                    turns += 1
            else:
                multi = 1
            val = self._multiplicator.value * multi**turns
            
            if 1 >= val >= 1E-24:
                # print(multi, turns)
                val = self._multiplicator.value * multi**turns
                if val == 1.0000000000000002e-15:
                    val = 1e-15
                elif val == 1.0000000000000002e-12:
                    val = 1e-12
                elif val == 1.0000000000000002e-09:
                    val = 1e-9
                elif val == 1.0000000000000002e-06:
                    val = 1e-6
                elif val == 1.0000000000000002e-03:
                    val = 1e-3
                name = TimeMultiplicator(TimeMultiplicator(val))
                if name.name != "s":
                    name = f'{name.name}s'
                else:
                    name = "s"
            elif self._multiplicator.value == 1.0 and value > 999.9 or val > 1:
                if val > 1:
                    inter = (inter * val)/60
                else:
                    inter = value/60
                if inter > 999.9:
                    inter = value/3600
                    name = "hours"
                else:
                    name = "mins"  
            else:
                msg = "time units are smaller than 1E-24, this cannot be handle"
                raise ExperimentException(msg)
        formatted = Decimal(inter)
        if negative:
            return f"-{str(formatted.quantize(Decimal(10) ** - decimal))} {name}"
        else:
            return f"{str(formatted.quantize(Decimal(10) ** - decimal))} {name}"

def get_root_directory():
    return pathlib.Path(__file__).parent.parent.parent

def select_traces(data, wavelength=None, space=10, points=1, avoid_regions=None):
    """ select traces in the wavelength range
    
    Parameters
    ----------
    data: ndarray
        Normally in  the original data set the number of rows should be the
        time points and the number of columns the Columns points. Although
        this function works with transposes data sets
            e.g.: data.shape = (50,250)
                  data has 250 wavelength points and 50 time points

    wavelength: 1darray or None
        Wavelength vectors where traces are selected. The len(wavelength),
        should be equal to the number of columns of data
        
        If None an array from 0 to data number of columns is created and
        returned after cutting. The parameters space should be given
        according to indexes
            
    space: int or list or "auto" (default 10)
        If type(space) = int: a series of traces separated by the value
        indicated will be selected.
        If type(space) = list: the traces in the list will be selected.
        If space = auto, the number of returned traces is 10 and equally spaced
        along the wavelength vector and points is set to 0
        
    points: int (default 1)
        binning points surrounding the selected wavelengths.
        e. g.: if point is 1 trace = mean(index-1, index, index+1)
    
    avoid_regions: list of list (default None)
        Defines wavelength regions that are avoided in the selection when space
        is an integer. The sub_list should have two elements defining the region
        to avoid in wavelength values
        i. e.: [[380,450],[520,530] traces with wavelength values between
                380-450 and 520-530 will not be selected
               
    Returns 
    ----------
    2d array with a selection of data traces, and 1darray with the corresponding 
    wavelength values
    
    """
    dat = pd.DataFrame(data)
    if type(space) == int or type(space) == list or space == 'auto':
        if wavelength is None:
            wavelengths = pd.Series([float(i) for i in range(dat.shape[1])])
        else:
            wavelengths = pd.Series(wavelength)
        if space == 'auto':
            number = round(len(wavelength) / 11)
            values = [i for i in range(len(wavelength))[::number]]
            selected_traces = values[1:]
            points = 0
        elif type(space) is int:
            if wavelength is not None:
                wavelength_unit = (wavelength[-1] - wavelength[0]) / len(wavelength)
                # print(wavelength_unit)
                # if wavelength_unit >= 1:
                #    space = round(space * wavelength_unit)
                #    print("aa")
            first = wavelengths.iloc[0 + points]
            values = [first + space * i for i in range(len(wavelengths)) if first + space * i < wavelengths.iloc[-1]]
            selected_traces = [(wavelengths - values[i]).abs().sort_values().index[0] for i in range(len(values))]
            # print(len(selected_traces))
        else:
            selected_traces = [np.argmin(abs(wavelengths.values - i)) for i in space]
        avoid_regions_index = []
        if avoid_regions is not None:
            msg = 'Please regions should be indicated as a list'
            assert type(avoid_regions) is list, msg
            if type(avoid_regions[0]) is not list:
                avoid_regions = [avoid_regions]
            for i in avoid_regions:
                msg = 'Please indicate 2 number to declare a region'
                assert len(i) == 2, msg
                i = sorted(i)
                avoid_wavelength = np.where((wavelength > i[0]) & (wavelength < i[1]))[0]
                if len(avoid_wavelength) > 0:
                    avoid_regions_index.append([avoid_wavelength[0], avoid_wavelength[-1]])
                selected_traces = [i for i in selected_traces if i not in avoid_wavelength]

        selected_traces = list(set(selected_traces))
        selected_traces.sort()

        if avoid_regions is None:
            if points == 0:
                dat_res = pd.DataFrame(data=[dat.iloc[:, i] for i in selected_traces],
                                       columns=dat.index,
                                       index=[str(i + wavelengths[0]) for i in selected_traces]).transpose()
            else:
                if type(space) is list:
                    indexes = []
                    for i in selected_traces:
                        mini = 0 if i - points < 0 else i - points
                        maxi = len(wavelengths) if i + points > len(wavelengths) else i + points + 1
                        indexes.append([mini, maxi])
                    dat_res = pd.DataFrame(data=[dat.iloc[:, i[0]:i[1]].mean(axis=1) for i in indexes],
                                           columns=dat.index,
                                           index=[str(i + wavelengths[0]) for i in selected_traces]).transpose()
                else:
                    dat_res = pd.DataFrame(
                        data=[dat.iloc[:, i - points:i + points + 1].mean(axis=1) for i in selected_traces],
                        columns=dat.index,
                        index=[str(i + wavelengths[0]) for i in selected_traces]).transpose()
            wavelength_res = np.array([wavelengths.iloc[i] for i in selected_traces])
        else:
            min_indexes = []
            max_indexes = []
            for trace in selected_traces:
                min_index = [sub_region[1] if sub_region[0] < trace - points < sub_region[1] else
                             trace - points for sub_region in avoid_regions_index]
                if min_index[0] < 0:
                    min_index[0] = 0
                min_indexes.append(min_index[0])
                max_index = [sub_region[0] if sub_region[0] < trace + points < sub_region[1] else
                             trace + points for sub_region in avoid_regions_index]
                max_indexes.append(min(max_index))
            dat_res = pd.DataFrame(data=[dat.iloc[:, min_index:max_index + 1].mean(axis=1) for min_index, max_index in
                                         zip(min_indexes, max_indexes)], columns=dat.index,
                                   index=[str(i + wavelengths[0]) for i in selected_traces]).transpose()
            wavelength_res = np.array([wavelengths.iloc[min_index:max_index + 1].mean() for min_index, max_index in
                                       zip(min_indexes, max_indexes)])

        return dat_res.values, wavelength_res
    else:
        statement_3 = 'space should be: "auto", an integer or a list of integers/floats'
        raise ExperimentException(statement_3)


def define_weights(time, rango, typo='constant', val=5):
    """
    Returns a an array that can be apply  in global fit functions as weights.
    The weights can be use to define areas where the minimizing functions is
    not reaching a good results, or to define areas that are more important
    than others in the fit. The fit with weights can be inspect as any other
    fit with the residual plot. A small constant value is generally enough
    to achieve the desire goal.

    Parameters
    ----------
    time: 1darray or None
        time vectors. the weight vector will have the same length w

    rango: list (length 2)
        list containing initial and final time values of the range
        where the weights will be applied

    typo: str (constant, exponential, r_exponential or exp_mix)
        defines the type of weighting vector returned

        constant: constant weighting value in the range
        exponential: the weighting value increase exponentially
        r_exponential: the weighting value decrease exponentially
        mix_exp: the weighting value increase and then decrease exponentially

        example:
        ----------
            constant value 5, [1,1,1,1,...5,5,5,5,5,....1,1,1,1,1]
            exponential for val= 2 [1,1,1,1,....2,4,9,16,25,....,1,1,1,]
                    for val= 3 [1,1,1,1,....3,8,27,64,125,....,1,1,1,]
            r_exponential [1,1,1,1,...25,16,9,4,2,...1,1,1,]
            exp_mix [1,1,1,1,...2,4,9,4,2,...1,1,1,]

    val: int (default 5)
        value for defining the weights

    Returns
    ----------
    a dictionary with the keys: parameters pass plus "apply": True
    and "vector": weighting vector. The dictionary can be used as kwargs
    in the global fitting function. Notice that only if apply is set
    to True then the fitting will consider the weights vector.
    """

    time = time * 1.0
    rango = sorted(rango)
    if typo in ['constant', 'exponential', 'r_exponential', 'mix_exp']:
        if typo == 'constant':
            weight = [val if rango[0] < i < rango[1] else 1 for i in time]
        else:
            mini = int(np.argmin([abs(i - rango[0]) for i in time]))
            maxi = int(np.argmin([abs(i - rango[1]) for i in time]))
            if typo == 'exponential':
                weight = [1 for i in time[:mini]] + [i ** val for i in range(
                    1, maxi - mini + 2)] + [1 for i in time[maxi + 1:]]
                weight[mini] = val
            elif typo == 'r_exponential':
                weight = [1 for i in time[:mini]] + [i ** val for i in range(
                    maxi - mini + 1, 1, -1)] + [1 for i in time[maxi:]]
                weight[maxi] = val
            else:
                if (maxi - mini) % 2 == 0:
                    weight = [1 for i in time[:mini]] + [i ** val for i in range(
                        1, (maxi - mini + 2) // 2)] + [i ** 2 for i in range(
                                                       (maxi - mini + 2) // 2, 1, -1)] + [1 for i in time[maxi:]]
                else:
                    weight = [1 for i in time[:mini]] + [i ** val for i in range(
                        1, (maxi - mini + 3) // 2)] + [i ** 2 for i in range(
                                                       (maxi - mini + 2) // 2, 1, -1)] + [1 for i in time[maxi:]]
                weight[mini] = val
                weight[maxi] = val
        return {'apply': True, 'vector': np.array(weight),
                'type': typo, 'range': rango, 'value': val}
    else:
        statement_3 = 'typo should be: constant, exponential, r_exponential or  mix_exp'
        raise ExperimentException(statement_3)


def read_data(path, wavelength=0, time=0, wave_is_row=False, separator=',', decimal='.'):
    """
    Read a data file from the indicated path and returns three arrays with shapes
    uses in the ultrafast. The function is bases in pandas read_csv, and uses
    the ReadData class.

    For the rows or columns corresponding to the wavelength and time vectors
    the function deals with non numerical values such as units names (e.g.: 'nm'
    'ps', 'ns'), words and number in scientific notation E or e in any of the forms
    and combinations. For the rest of values (data values) the function assumes they
    are homogeneous. The function eliminates from the data rows or columns with non
    numerical numbers (Nan) in all the entries if not they are set to zero. It also sort
    the time points if these where non sorted as can be the case with some multi-probe
    experimental outputs data files.

    The arrays return are:
        time: 1darray of length equal to the number of rows in data

        data: 2darray, where rows correspond to the time direction
              and wavelength to the columns

        wavelength: 1darray of length equal to the number of columns
                    in data

    Parameters
    ----------
    path: str
        path to the data file

    wavelength: int (default 0)
        defines the element where to find the wavelength vector in its direction
        which is defined by wave_is_row parameter. i.e.: if wavelength correspond
        to columns, wavelength=0 indicates is the first column of the data file
        if wavelength correspond to rows, then wavelength=0 is first row


    time: int (default 0)
        defines the element where to find the time vector in its direction
        which is defined by wave_is_row parameter. i.e.: if times correspond
        to columns, time=0 indicates is the first column of the data file

    wave_is_row: bool (default False)
        defines if in the original data set the wavelength correspond
        to the rows.

    separator: str (default ',')
        defines the separator in the data (any value that can be used in pandas
        read_csv is valid. For tab uses \t

    decimal: int (default '.')
        defines the decimal point in the data

    Returns
    ----------
    3 arrays corresponding to time, data, and wavelength
    """
    reader = ReadData()
    time, data, wavelength = reader.readData(path,
                                             wavelength=wavelength,
                                             time=time,
                                             wave_is_row=wave_is_row,
                                             separator=separator,
                                             decimal=decimal)
    return time, data, wavelength


class ReadData:
    @staticmethod
    def _readPandas(pandas):
        """
        return the index and columns names of a pandas data frame as numpy
        arrays if this are form by numbers or a number plus a string. For
        example if for a dataFrame with index or columns as [4 nm, 5 nm, 6 nm]
        is returns [4, 5, 6]
        """
        try:
            column = np.array([float(i) for i in pandas.columns.values])
        except:
            column = np.array([float((re.findall(r'[-+]?\d*\.\d*[eE]?[-+]?\d*|[-+]?\d+', i))[0]) for i in
                               pandas.columns.values]).flatten()
        if type(pandas.index[0]) == str:
            row = np.array([float((re.findall(r'[-+]?\d*\.\d*[eE]?[-+]?\d*|[-+]?\d+', i))[0]) for i in
                            pandas.index.values]).flatten()
        else:
            row = np.array([float(ii) for ii in pandas.index.values])
        return row, column

    def readData(self, path, wavelength=0, time=0, wave_is_row=True,
                 separator=',', decimal='.'):
        """
        similar parameters and explanations as in "read_data" function
        """
        if wave_is_row:
            # print('row')
            data_frame = pd.read_csv(path, sep=separator, index_col=time,
                                     skiprows=wavelength, decimal=decimal).dropna(
                                     how='all', axis=1).dropna(how='all', axis=1)
            data_frame = data_frame.transpose()
        else:
            data_frame = pd.read_csv(path, sep=separator, index_col=wavelength,
                                     skiprows=time, decimal=decimal).dropna(
                how='all').dropna(how='all', axis=1)
        data_frame.fillna(0, inplace=True)
        wavelength_dimension, time_dimension = ReadData._readPandas(data_frame)
        if wavelength_dimension[0] > wavelength_dimension[-1]:
            wavelength_dimension = wavelength_dimension[::-1]
            data_frame = data_frame.iloc[::-1, :]
        time_dimension = sorted(time_dimension)
        data_frame.reindex(time_dimension).sort_index()
        return np.array(time_dimension), data_frame.transpose().values, wavelength_dimension


def solve_kmatrix(exp_no, params):
    """
    Resolve the k_matrix from a parameters object for a target fit

    Parameters
    ----------
    exp_no: int
        number of component of the K_matrix (diagonal elements)

    params: lmfit parameters object
        object containing the parameters
    Returns
    ----------
    Coefficients of each component eigenvalues and eigen matrix
    """
    ksize = exp_no
    kmatrix = np.array(
        [[params['k_%i%i' % (i + 1, j + 1)].value for j in range(ksize)]
         for i in range(ksize)])
    cinitials = [params['c_%i' % (i + 1)].value for i in range(ksize)]
    # do the eigens value decomposition
    eigs, vects = np.linalg.eig(kmatrix)
    # eigenmatrix = np.array([[vects[j][i] for j in range(len(eigs))] for i in range(len(eigs))])
    eigenmatrix = np.array(vects)
    coeffs = np.linalg.solve(eigenmatrix, cinitials)
    return coeffs, eigs, eigenmatrix


def book_annotate_all_methods(book=None, cls=None):
    if book is None:
        book = LabBook()
    if cls is None:
        return lambda cls: book_annotate_all_methods(book, cls)

    class DecoratedClass(cls):
        if hasattr(cls, "book"):
            obj = getattr(cls, "book")
            if isinstance(obj, LabBook):
                pass
            else:
                cls.book_annotate = book
        else:
            cls.book = book

        def __init__(self, *args, **kargs):
            super().__init__(*args, **kargs)

        def __getattribute__(self, item):
            value = object.__getattribute__(self, item)
            if callable(value):
                return book_annotate(cls.book)(value)
            return value

    return DecoratedClass


def froze_it(cls):
    cls.__frozen = False
    cls.__modified = False

    def frozen_setattr(self, key, value, code=None):
        """
        Function that allow to only set an attribute it is new. It can be
        modified only if the code is correct.
        """
        val = 1
        if code is not None:
            x = np.random.RandomState(code)
            val = x.randint(10000, size=1)[0]

        if self.__frozen and hasattr(self, key) and val != 2603:
            if val != 1:
                print("INCORRECT CODE: Contact creator for more information")
            print("Class {} is frozen. Cannot modified {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)
            if val == 2603:
                print("CORRECT CODE: Attribute modified")
                object.__setattr__(self, '__modified', True)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = frozen_setattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


# @froze_it
# class UnvariableContainer:
#     """
#     Object where once an attribute has been set cannot be modified if
#     self.__frozen = False
#     """
#     def __init__(self, **kws):
#         for key, val in kws.items():
#             setattr(self, key, val)


def book_annotate(container, extend=True):
    """
    Decorator with arguments. This decorator is working only with LabBook class,
    or an object with modified __setattr__ method.

    It is use to create a lab book where the functions names and parameters
    names and values given are saved in container (VariableContainer class).
    VariableContainer has a method print() for printing the lab book.
    """

    def dump_args(func):
        """
        This decorator dumps out the arguments passed to a function before
        calling it
        """
        argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
        if argnames[0] == 'self':
            argnames = argnames[1:]
        fname = func.__name__
        
        # @wraps use to keep meta data of func
        @wraps(func)
        def echo_func(*args, **kwargs):
            valores = dict(zip(argnames, args), **kwargs)
            if func.__defaults__ is not None:
                defaults = dict(zip(argnames[-len(func.__defaults__):],
                                    func.__defaults__))
                for i in defaults.keys():
                    if i not in valores.keys():
                        valores[i] = defaults[i]
            try:
                result = func(*args, **kwargs)
            except Exception as exception:
                traceback.print_exc()
                return exception.__str__()
            else:

                container.__setattr__(fname,
                                      ', '.join('%s = %r' % entry
                                                for entry in valores.items()),
                                      extend)
                return result

        return echo_func

    return dump_args


class LabBook(object):
    """
    Class object that can store in combination with book_annotate the
    functions ran names as attributes where the values are strings with
    the function parameters names and the values passed.
    """

    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)
        self.creation = datetime.datetime.now().strftime("day: %d %b %Y | hour: %H:%M:%S")

    def __setattr__(self, key, val, extend=False):
        """
        setter modification to include an extend parameter working with
        book_annotate decorator

        Parameters
        ----------
            extend: bool (default True)
                If True, and a same function is run several times, then the
                records are keep as a list of string.
                If False, if a same function is run twice the previous meta data
                will be overwritten.
        """
        if hasattr(self, key) and extend:
            prev_val = getattr(self, key)
            if type(prev_val) != list:
                prev_val = [prev_val]
            prev_val.append(val)
            super().__setattr__(key, prev_val)
        else:
            super().__setattr__(key, val)
    
    @property
    def actions(self):
        actions = [i for i in self.__dict__.keys() if i not in ["name", "creation"] and i[0] != '_']
        return actions

    @property
    def protected_actions(self):
        actions = [i for i in self.__dict__.keys() if i not in ["name", "creation"] and i[0] == '_']
        return actions

    def clean(self):
        """
        Clean the LabBook except name attribute if given
        """
        for key, value in self.__dict__.items():
            if key != 'name' and key != 'creation':
                self.delete(key)
        self.creation = datetime.datetime.now().strftime("day: %d %b %Y | hour: %H:%M:%S")

    def delete(self, key, element='all'):
        """
        method similar to delattr function. It deletes an entry on the lab
        according to key

        Parameters
        ----------
            key: str
                name of the attribute

            element: "all" or int (default "all")
                if "all" all entries will be deleted
                if int the element of the list at this index is deleted
        """
        prev_val = getattr(self, key)
        if element == 'all' or type(prev_val) != list:
            delattr(self, key)
        else:
            if type(element) == int:
                prev_val.pop(element)
                setattr(self, key, prev_val)
            else:
                statement = 'element should be "all" or a integer'
                raise ExperimentException(statement)

    def print(self, creation=True, print_protected=False, single_line=False):
        """
        Print all attributes and their values as a Lab report

        Parameters
        ----------
            creation: bool (default True)
                If True prints the day and hour of the LabBook creation
                
            print_protected: bool (default False)
                Define if protected attributes starting with "_" are printed

            single_line: bool (default False)
                If True attributes that are not a list, the name and value will
                be printed in a single line
        """
        to_print = self.__str__(creation=creation,
                                print_protected=print_protected,
                                single_line=single_line)
        print(to_print)

    def __str__(self, creation=True, print_protected=False, single_line=False):
        """
        Allows to do print(LabBook)
        """
        to_print = []
        if hasattr(self, 'name'):
            name = getattr(self, 'name')
            to_print.append(f'\t {name}')
            to_print.append(''.join(['-' for i in range(len(name) + 10)]))
        for key, value in self.__dict__.items():
            if key != 'notes' and key != 'name' and key != 'creation' and \
                    key[0] != "_":
                to_print.append(self._print_attribute(key,
                                                      single_line,
                                                      False))
        if hasattr(self, 'notes'):
            to_print.append(self._print_attribute('notes', False, False))
        if print_protected:
            for key, value in self.__dict__.items():
                if key[0] == "_" and key[1] != "_":
                    to_print.append(self._print_attribute(key,
                                                          single_line,
                                                          True))
        if creation:
            to_print.append(self._print_attribute('creation', True, False))
        return "\n".join(to_print)

    def _print_attribute(self, key, single_line=True, protected=True):
        """
        print single attribute
        """
        to_print = []
        value = getattr(self, key)
        if protected:
            val = f'(p) {key}'
        else:
            val = ' '.join(key.split('_'))
        if type(value) == list:
            to_print.append(f'\t {val}:')
            for i in value:
                to_print.append(f'\t\t {i}')
        else:
            if single_line:
                to_print.append(f'\t {val}: {value}')
            else:
                to_print.append(f'\t {val}:\n\t\t {value}')
        to_print.append('')
        return "\n".join(to_print)


@froze_it
class UnvariableContainer(LabBook):
    """
    Object where once an attribute has been set cannot be modified if
    self.__frozen = True
    """
    def __init__(self, **kws):
        super().__init__(**kws)

    def __delattr__(self, name):
        print(f"The class UnvariableContainer is frozen and attribute"
              f" '{name}' cannot be deleted")


class FiguresFormating:
    """
    Class containing static methods for axis matplotlib formatting
    """

    @staticmethod
    def cover_excitation(ax, x_range, x_vector):
        """
        add a white rectangle on top of an area in the figure. Typically use to
        cover the excitation of the laser in the spectra figure of a pump-probe
        UV-vis experiment.

        Parameters
        ----------
        ax: matplotlib axis
            axis containing the figure

        x_range: list or tupple of length 2
            contains the initial and and final x values of x vector
            (normally wavelength)

        x_vector: array
            the x vector plotted in the figure
        """
        ymin, ymax = ax.get_ylim()
        ymin = ymin - ymin * 0.05
        ymax = ymax - ymax * 0.05
        mini = np.argmin([abs(x_range[0] - i) for i in x_vector])
        maxi = np.argmin([abs(x_range[1] - i) for i in x_vector])
        rect = Rectangle((x_vector[mini] - 1, ymin),
                         width=x_vector[maxi] - x_vector[mini] + 2,
                         height=abs(ymax) + abs(ymin),
                         fill=True, color='white', zorder=np.inf)
        ax.add_patch(rect)

    @staticmethod
    def axis_labels(ax, x_label=None, y_label=None):
        """
        add labels to the axis ax

        Parameters
        ----------
        ax: matplotlib axis
            axis containing the figure

        x_label: str (default None)
            string containing the x label,

        y_label: str (default None)
            string containing the y label
        """
        if x_label is None:
            x_label = 'X vector'
        if y_label is None:
            y_label = 'Y vector'
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    @staticmethod
    def format_figure(ax, data, x_vector, size=14, x_tight=False,
                      set_ylim=True, val=50):
        """
        Deprecated function, style in combination with use_style decorator
        should be use to format figures.

        format ax figures
        Always does:
            -add minor tick
            -set ticks directions in
            -add tick in top and left sides
            -set tick labels in scientific notations
            -add x = 0 horizontal line. line_style = (--)

        Set margins of the data figure in x and y directions according to the
        different parameters.

        Parameters
        ----------
        ax: matplotlib axis
            axis containing the figure

        data: np array
            array containing the data plotted. Is used to get minimum and
            maximum values and adjust the y margins if set_ylim is True

        x_vector: array
            the x vector plotted in the figure

        size: int
            size of the the tick labels

        x_tight: bool (default False)
            If True, set tight the x axis
            If False set limits according to max(x_vector)/val

        val: int or float
            value for adjusting x vector if x_tight is False
        """
        if val < 1:
            val = 1
        if set_ylim:
            ax.set_ylim(np.min(data) - abs(np.min(data) * 0.1), np.max(data) + np.max(data) * 0.1)
        if x_tight:
            ax.set_xlim(x_vector[0], x_vector[-1])
        else:
            ax.set_xlim(x_vector[0] - x_vector[-1] / val, x_vector[-1] + x_vector[-1] / val)
        ax.axhline(linewidth=1, linestyle='--', color='k')
        ax.ticklabel_format(style='sci', axis='y')
        ax.minorticks_on()
        ax.axes.tick_params(which='both', direction='in', top=True, right=True, labelsize=size)
        msg = 'Deprecated function, style in combination with use_style ' \
              'decorator should be use to format figures.'
        print(msg)


class DataSetCreator:
    """
    Class that with static methods that can be use to generate examples
    of transient data, considering gaussian spectral shapes shapes. The
    output is a pandas dataFrame.
    example how to use (from DAS):
    -------------------
    wave = DataSetCreator.generate_wavelength(350,700,500)
    time = DataSetCreator.generate_time(-2,500,120)
    shapes = DataSetCreator.generate_shape(5,wave, taus=[8, 30, 200], signe = 0, sigma = 50)
    data_set = DataSetCreator.generate_dataset(shapes, time, 0.12)
    After fitting the decay associated spectra should be equal to shapes, and the
    fitting times 8 30 and 200.
    example how to use (from SAS):
    -------------------
    wave = DataSetCreator.generate_wavelength(400,700,500)
    peaks=[[470,550,650],[480,700],[500,600]]
    amplitudes=[[-10,-6,12],[-9,11],[-8,6]]
    fwhms=[[50,100,80],[40,70],[30,65]]
    shapes = DataSetCreator.generate_specific_shape(wave, taus=3, peaks=peaks, amplitudes=amplitudes, fwhms=fwhms)
    k1=1/5.0
    k2=1/20.0
    k3=1/100.0
    kmatrix = [[-k1,0,0],[k1/3,-k2,0],[2*k1/3,0,-k3]]
    initials = [1.0,0,0]
    profiles = DataSetCreator.generate_profiles(500.0,5000,initials,kmatrix)
    data_set_conv = DataSetCreator.generate_dataset(shapes, profiles, 1.2)
    new_times = DataSetCreator.generate_time(data_set_conv.index[0],data_set_conv.index[-1],120)
    data_set_conv_proj = DataSetCreator.timegrid_projection(data_set_conv, new_times)
    Above kmatrix represent model where species 1 bifurcates to species 2 and 3 (with 1/3 and 2/3 probablity, respectively),
    and each of them decay independently with 1/k2 and 1/k3.
    initials = [1/3.0, 1/3.0, 1/3.0]
    kmatrix = [[-k1,0,0],[0,-k2,0],[0,0,-k3]]
    Above kmatrix and initial conditions should generate just 3 exp decay (DAS equivalent)
    initials = [1.0,0,0]
    kmatrix = [[-k1,0,0],[k1,-k2,0],[0,k2,-k3]]
    Above kmatrix and initial conditions should generate sequential model (cascade, 1->2->3, EAS equivalent)
    Some comments about how to build kmatrix. It is just a matrix of rates, which when multiplied by concentration
    vector of given species results in derrivative of concentration vector of the same/other species by time.
    In other words, dc_i/dt = sum of k_ij * c_j . In general derrivative of vector of c-s = kmatrix * vector of c-s.
    So for example, first row of kmatrix are k-s of species which decay to first species (positive values),
    except first one (k[0,0]) which must be negative because it describes decay of species one.
    (1) diagonal k-s must be negative, because they describe decay of given species,
    (2) If you sum all kij*cj parts in all rows, it should yield zero for the sum of populations to be preserved.
    If this condition is not satisfied, then it is still okay, assuming that there is more negative values and you
    are aware with the fact that your sum of c-s decays to zero (typical to TR spectroscopy).
    (3) Firstly you can start with defining diagonal elements, which are negative reciprocals of lifetimes
    of given species
    (4) Then You decide to which species each species decay, and you balance this negative diagonal k in other
    species which has lower energy in energy ladder of states.
    (5) You can skip point 4 and just let given species decay to nothing (usually GS) or you can also divide
    any diagonal k to some portions in more than one different species. Then you have splitting (bifurcation),
    so one species decay to more than one species. In general it means, that in each column of kmatrix sum of
    k-s must be zero (concentration preserved over time) or negative (concentration goes down, so species decay
    to GS and TA signal goes to zero). After fitting the data parameters should reproduce model described by kmatrix.
    This class does not use graphical model generator, because its intention is to test all fitting code, so everything
    need to be implemented separately, and possibly in a different way to avoid mistake-compensation.
    There are still some minor problems (odeint function gives sometimes NaNs at output, but works if you run it again,
    so it seems that some SAS inputs are crashing it, needs to be checked),nevertheless usually works ok and gives
    reasonable results.

    Comment: It seems that when using generate_specific_shape instead of generate_shape there is no errors!
    """

    @staticmethod
    def gauss(x, amp, cen, sigma):
        """
        basic gaussian function
        """
        return amp * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))

    @staticmethod
    def norm_gauss(x, fwhm, x0=0.0):
        """
        just normal (area=1) gaussian distribution
        """
        sigma = fwhm / 2.355
        return 1 / (sigma * np.sqrt(2 * np.pi)) * \
            np.exp(-(x - x0) * (x - x0) / (2 * sigma * sigma))

    @staticmethod
    def generate_shape(number, wave, taus, scale=100, signe=1, sigma=2.25):
        """
        !WARNING: Sometimes this function generates NaNs instead values. Should be checked.
        function to generate the initial shape (DAS) of the data set. The number of taus
        defines the shape of the output. returns a pandasDataFrame where the index are
        taus and the columns the wavelength. The parameters of the curves are ramdomly
        generated.
        Parameters
        ----------
        number: int
            defines number of maxima/minima int the output shape. Should be greater than len(taus).
            e.g.: if number is 5 and taus=[8, 30, 200]. The output is a three row dataFrame, where
            two rows have two maxima/minima and the third only one (2+2+1 = 5)
        wave: np array
            array containing the wavelength vector
        taus: list (for example [8, 30, 200]) or int
            list containing the associated decay times of the data
            or number of DAS, but then DAS won't be labelled by taus, still can be used as SAS
        scale: int (default 100)
            controls the intensity of the output data
        signe: -1, 0 1
            Defines the if the curve is positive or negative
            1: all positive
            0: can be positive or negative
            -1: all negative
        sigma:
            default base sigma value of the Gaussian curves (The final is randomly attribute)
        Returns
        ----------
        pandas data frame.  data.shape() >>> (len(taus), len(wave))
        """
        if(type(taus) == int):  # useful for target model usage, because then tau values are useless, only number of species is required
            taus = [x + 1 for x in range(taus)]

        if number >= len(taus):
            if signe == 1:
                a, b = 0, 9
            elif signe == -1:
                a, b = -9, 0
            elif signe == 0:
                a, b = -9, 9
            else:
                raise ExperimentException('signe should be either -1, 0, 1')
            rango = (wave[-1] - wave[0] - 100) / 2
            gausianas = number
            # generate random values for N (number) gausians
            amp = [
                0.60 +
                0.50 *
                np.random.randint(
                    a,
                    b) for i in range(gausianas)]
            cen = [wave[0] + 50 + i * rango /
                   (number * 0.5) for i in range(gausianas)]
            sig = [sigma + 6 * np.random.rand() for i in range(gausianas)]
            # create gaussians with params
            datag = [(DataSetCreator.gauss(wave, amp, cen, sig))
                     for amp, cen, sig in zip(amp, cen, sig)]
            # divide by a number to adjust to scale
            gaus = [i / scale for i in datag]
            if number > len(taus):
                tauss = [np.random.choice(taus, replace=False) for i in range(len(taus))] + \
                        [np.random.choice(taus) for i in range(gausianas - len(taus))]
            else:
                tauss = [
                    np.random.choice(
                        taus,
                        replace=False) for i in range(
                        len(taus))]
            das = np.ones((len(taus), len(wave)))
            for i, tau in enumerate(taus):
                tau1 = np.mean([ii for i, ii in enumerate(
                    gaus) if tauss[i] == tau], axis=0)
                das[i, :] = tau1
            das = pd.DataFrame(data=das, columns=[str(
                round(i, 1)) for i in wave], index=taus)
            return das
        else:
            raise ExperimentException(
                'number should be >= than the number of taus')

    @staticmethod
    def generate_specific_shape(wave, taus, peaks, amplitudes, fwhms):
        """
        Alternative function to generate the initial shape (DAS or SAS) of the data set.
        This is not-random replacement for generate_shape, where you specify every propery of spectra.
        Returns a pandasDataFrame where the index are taus or species number and the columns are the wavelength.
        Parameters
        ----------
        wave: np array
            array containing the wavelength vector
        taus: list (for example [8, 30, 200]) or int
            list containing the associated decay times of the data
            or number of DAS, but then DAS won't be labelled by taus, still can be used as SAS
        peaks: 2D array / list of lists of floats
            Defines positions of the peaks in generated spectra. First dimension enumerates number of spectrum and
            must be aligned with taus number/size. Second dimension dspecifies number of peak in given spectrum.
            Different spectra can have different number of peaks inside.
        ampitudes: 2D array / list of lists of floats
            The same as above, but specifies amplitudes of peaks (nonzero, positive or negative).
            Note that peaks are normal gauss, so area under will be abs(peaks[i][i])*1.0
        fwhms: 2D array / list of lists of floats
            The same as above, but specifies FWHMs (widths) of peaks (values must be positive).
        Returns
        ----------
        pandas data frame.  data.shape() >>> (len(taus), len(wave))
        """
        # useful for target model usage, because then tau values are useless, only number of species is required
        if type(taus) == int:
            taus = [x + 1 for x in range(taus)]

        if len(taus) != len(peaks) or len(peaks) != len(amplitudes) or len(amplitudes) != len(fwhms):
            raise ExperimentException(
                'Check if taus/peaks/amplitudes/fwhms have the same dimension and represent same number of DAS/SAS!')

        das = np.zeros([len(taus), len(wave)])

        for i in range(len(taus)):
            if len(peaks[i]) != len(amplitudes[i]) or len(amplitudes[i]) != len(fwhms[i]):
                raise ExperimentException(
                    'Check if peaks/amplitudes/fwhms[' +
                    str(i) +
                    '] have the same second dimension and represent same number of peaks!')
            for j in range(len(peaks[i])):
                if fwhms[i][j] <= 0.0:
                    raise ExperimentException(
                        'Check if all FWHMs are positive!')
                das[i, :] += amplitudes[i][j] * \
                    DataSetCreator.norm_gauss(wave, fwhms[i][j], x0=peaks[i][j])

        return pd.DataFrame(data=das, columns=[str(
            round(i, 1)) for i in wave], index=taus)

    @staticmethod
    def generate_wavelength(init, final, points):
        """
        returns a positive vector that should be use as wavelength. The points
        are equally spaced.
        Parameters
        ----------
        init: int
            initial value of the output vector
        final: np array
            final value of the output vector
        points: int
            length of output the array
        returns
        -------
        1d array (positive)
        """
        return np.linspace(abs(init), abs(final), points)

    @staticmethod
    def generate_time(init, final, points, space='lin-log'):
        """
        returns a vector that should be use as time. The points
        can be linear logarithmic or in a combined spaced way.
        Parameters
        ----------
        init: int
            initial value of the output vector
        final: np array
            final value of the output vector
        points: int
            length of output the array
        space: str; lin-log, linear or log (default lin-log)
            linear: points are linearly spaced
            log: points are spaced log10 scale
            lin-log: linearly spaced for values < 1 and log10 spaced for values > 1
        returns
        -------
        1d array
        """
        if space in ['linear', 'log', 'lin-log']:
            if space == 'lin-log':
                if init > 1:
                    space = 'log'
                else:
                    return np.append(np.linspace(init, 1, round(points / 3) + 1)[:-1],
                                     np.logspace(np.log10(1), np.log10(final),
                                                 round(2 * points / 3)))
            elif space == 'log':
                if init < 1:
                    return np.logspace(0, np.log10(final + abs(init) + 1),
                                       points) + init - 1
                else:
                    return np.logspace(np.log10(init),
                                       np.log10(final + abs(init)), points)
            else:
                return np.linspace(init, final, points)
        else:
            raise ExperimentException(
                'space should be either "linear" "log" or "lin-log"')

    @staticmethod
    def generate_dataset(shape, time, fwhm=None):
        """
        generate the data set from the initial spectral shape form. The data has no noise. This
        can be added using the add_noise function.
        Parameters
        ----------
        shape: pandas DataFrame
            initial value of the output vector
        time: 1darray or pd.DataFrame
           time vector with delays will trigger generation from DAS (sums of exps)
           or pd.DataFrame generated by DataSetCreator.generate_profiles function will trigger generation from SAS
        fwhm: None or float
            if None the time evolution will start instantly at time zero
            if positive float value, evolution will be convolved with gaussian of given FWHM
        returns
        -------
        pandas data frame.  data.shape() >>> (len(time), shape.shape[1])
        """
        # generate dataset from SAS and concentration profiles (from model)
        if(type(time) == pd.DataFrame):
            # because profiles start with time=0, i need to add negative times due to gaussian IRF
            if(fwhm is not None):
                t_min = -6 * fwhm
                times = time.index.to_numpy()
                if times[0] != 0:
                    msg = "Timegrid for dataset generation from SAS should start from zero!"
                    raise ExperimentException(msg)
                # add some negative times which are reflection of positive ones
                addtimes = -times[1:np.argmin(np.abs(times + t_min))]
                addtimes_sorted = np.sort(addtimes)

                t_min_g = -3 * fwhm
                times_g = times[1:np.argmin(np.abs(times + t_min_g))]
                times_g_neg = np.sort(-times_g)
                # this is time grid around 0 used to build gauss function to do
                # convolution
                gauss_x = np.append(np.append(times_g_neg, [0.0]), times_g)
                gauss_y = DataSetCreator.norm_gauss(
                    gauss_x, fwhm)  # build gauss function for convolution

                newdf = pd.DataFrame(data=np.zeros(
                    [addtimes_sorted.shape[0], time.shape[1]]), index=addtimes_sorted, columns=time.columns)
                # now we have also negative time values with zero signal
                time = newdf.append(time)

                profile_x = time.index.to_numpy()

                gauss_len = gauss_x.shape[0]
                dx = (profile_x[5] - profile_x[0]) / 5

                new_profile_x = [profile_x[p + int((gauss_len - 1) / 2)]
                                 for p in range(0, profile_x.shape[0] - gauss_len + 1)]
                time_after_conv = pd.DataFrame(
                    index=new_profile_x, columns=time.columns)

                # this is not very fast, but does not have to be, and
                # bug-freedom is priority one. later we can speed up.
                for i in range(time.columns.shape[0]):
                    profile_y = time.values[:, i]
                    time_after_conv.values[:, i] = [np.trapz(gauss_y * profile_y[p:p + gauss_len], dx=dx)
                                                    for p in range(0, profile_x.shape[0] - gauss_len + 1)]
                time = time_after_conv

            time.columns = [i + 1 for i in range(time.shape[1])]
            shape.index = [i + 1 for i in range(shape.shape[0])]
            return time.dot(shape)
        else:  # generated dataset from DAS
            taus = shape.index
            values = [[[shape.values[i][ii], taus[i]] for i in range(
                shape.shape[0])] for ii in range(shape.shape[1])]
            if fwhm is None:
                spectra = [ModelCreator.expN(time, 0, time[0], values[i]) for i in range(len(values))]
            else:
                spectra = [ModelCreator.expNGauss(time, 0, 0, fwhm, values[i]) for i in range(len(values))]
            # convert to a pandas data frame
            spectra_final = pd.DataFrame(data=spectra, columns=[str(
                round(i, 4)) for i in time], index=shape.columns).dropna(axis=1)
            spectra_final = spectra_final.transpose()
            return spectra_final

    @staticmethod
    def timegrid_projection(input_data, time):
        """
        generate the data set from another dataset, with changed grid of delays
        designed to reduce the number and distribution of timepoints after generation from target model
        this is interpolation-only algorithm, no bucket-averaging!
        Parameters
        ----------
        input_data: pandas DataFrame
            generated dataset from DataSetCreator.generate_dataset
        time: 1darray or pd.DataFrame
           time vector with delays, which will be used to reproject the data, generated by DataSetCreator.generate_time
           must be within existing time grid of input_data (otherwise error will be thrown!)
        returns
        -------
        pandas DataFrame, just input with exchanged tme grid
        """
        old_time = input_data.index.to_numpy()
        if old_time[0] > time[0]:
            raise ExperimentException(
                "New timegrid must start with later delay than start of old timegrid!")
        if old_time[-1] < time[-1]:
            raise ExperimentException(
                "New timegrid must end with earlier delay than end of old timegrid!")
        # vector of closest points on old grid
        closest_p = [np.argmin(np.abs(old_time -time[x])) for x in range(time.shape[0])]
        # search for closest point on other side of new grid point
        opposite_p = closest_p + np.sign(time - old_time[closest_p])
        # if two point overlap, move one
        opposite_p = opposite_p + (np.abs(np.sign(opposite_p - closest_p)) - 1)
        if opposite_p[0] < 0:
            # except first, it needs to be moved foward not back
            opposite_p[0] + 2
        opposite_p = opposite_p.astype(int)

        # only argument dependent eq part
        eq_mult = (time - old_time[closest_p]) / \
            (old_time[closest_p] - old_time[opposite_p])

        old_kinetic = input_data.values
        new_kinetic = np.transpose(np.array([old_kinetic[closest_p, w] +
                                             (old_kinetic[closest_p, w] - old_kinetic[opposite_p, w]) * eq_mult
                                             for w in range(old_kinetic.shape[1])]))

        return pd.DataFrame(data=new_kinetic, columns=input_data.columns, index=time)

    @staticmethod
    def add_noise(data, scale=0.0005):
        """
        add gaussian random noise to the data
        Parameters
        ----------
        data: pandas DataFrame
            dataFrame without noise
        scale: float (default 0.0005)
           time vector
        returns
        -------
        pandas data frame equal size of data
        """
        spectra_final_noise = data * 0.0
        for i in range(spectra_final_noise.shape[0]):
            # add white noise to data
            spectra_final_noise.iloc[i, :] = data.iloc[i, :] + \
                np.random.normal(size=data.shape[0], scale=scale)
        return spectra_final_noise

    @staticmethod
    def derrivatives(cs, t, kmatrix):
        """
        computes derrivatives of concentrations over time, this is callback function for odeint
        Parameters
        ----------
        cs: numpy array 1D
            concentration of each population
        t: float
           given time point, does not contribute anyway
        kamtrix: numpy array 2D
           matrix of k-s, shape[1] must match shape[0] of cs (to properly perform product of matrixes)
        returns
        -------
        1D numpy array with shape[0] equal to cs.shape[0]. represents derrivatives of concentrations over time
        here is simple example:
        [[k_11, k_12] [[c_1]
        [k_21, k_22]] [c_2]]
        """
        return np.dot(kmatrix, np.reshape(cs, [cs.shape[0], 1]))[:, 0]

    @staticmethod
    def generate_profiles(final, points, initials, kmatrix):
        """
        SAS equivalent of DataSetCreator.generate_time function.
        It always starts from time zero and timegrid is linear (it has to be for odeint)
        Later one can project the data into another nonlinear grid, to reduce size of the data.
        Parameters
        ----------
        final: np array
            final value of the output vector
        points: int
            length of output the array
            important note: array must dense enough to cover well even short delays (where rapid changes take place),
            otherwise differential equations may be solved unaccurately. later one can see reproject time grid and reduce it to reasonable size
        initials: numpy array 1D
            initial (at time zero) concentration of each population
        kmatrix: numpy array 2D, dimensions equal to initials
           matrix of k-s describes evolution of the system
        returns
        -------
        1D numpy array with shape[0] equal to cs.shape[0]. represents derrivatives of concentrations over time
        here is simple example:
        [[k_11, k_12] [[c_1]
        [k_21, k_22]] [c_2]]
        """
        initials = np.array(initials)
        kmatrix = np.array(kmatrix)
        if kmatrix.shape[0] != kmatrix.shape[1] or kmatrix.shape[1] != initials.shape[0]:
            raise ExperimentException(
                'Check if kmatrix and initials have proper dimensions!')
        timegrid = np.linspace(0, final, points)
        profiles = odeint(DataSetCreator.derrivatives, initials, timegrid, args=(kmatrix,))
        return pd.DataFrame(data=profiles, columns=[str(i + 1) for i in range(initials.shape[0])],
                            index=timegrid)
