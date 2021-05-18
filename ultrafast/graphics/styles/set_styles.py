"""
Created on Thu Feb 25 14:31:39 2021

@author: lucas
"""
from pathlib import Path
import re
import os
from os.path import join, dirname, realpath
import json
from functools import wraps
import matplotlib.pyplot as plt
from ultrafast.graphics.styles.plot_base_functions import *

UTF_STYLE_DIR = dirname(__file__)
UTF_BASIC_STYLES = realpath(join(UTF_STYLE_DIR, 'basic_styles'))
STYLE_EXTENSION = 'mplstyle'
STYLE_FILE_PATTERN = re.compile(r'([\S]+).%s$' % STYLE_EXTENSION)

base = plt.style.core.load_base_library()
base_update = plt.style.core.update_user_library(base)
utf = plt.style.core.read_style_directory(UTF_BASIC_STYLES)

library = {**base_update, **utf}


def check_if_valid_style(name: str):
    """
    Return True if the name pass is a key of the matplotlib library containing
    all available styles
    """
    return True if name in library.keys() else False


def get_combined_style(styles):
    """
    Combined several matplotlib styles in a single combine style

    Parameters
    ----------
    styles: list
    List containing the names of the matplotlib styles to be combined
    """
    if type(styles) == str:
        if check_if_valid_style(styles):
            return library[styles]
        else:
            if isinstance(styles, (str, Path)):
                return styles
            else:
                msg = 'Not a valid ultrafast style'
                raise Exception(msg)
    else:
        mix_style = [library[i] for i in styles if check_if_valid_style(i)]
        if len(mix_style) > 0:
            style = {k: v for d in mix_style for k, v in d.items()}
            return style


class FigureStyle:
    STYLE_DIR = None

    def __init__(self, name):
        self.name = name

    def get_style(self):
        pass


class MplFigureStyle(FigureStyle):
    def get_style(self):
        check = check_if_valid_style(self.name)
        if check:
            return library[self.name], None, None
        else:
            return None


class UtfFigureStyle(FigureStyle):
    def get_style(self):
        data = self._get_file()
        if data is not None:
            styles = data["styles"]
            style = get_combined_style(styles)
            funct, funct_arg = self._get_utf_style_function(data)
            return style, funct, funct_arg
        else:
            return None

    def _get_file(self):
        val = self._get_file_name()
        if val is not None:
            path = realpath(join(UTF_STYLE_DIR, val))
            with open(path, 'r') as f:
                data = json.load(f)
                if "utf_style" in data.keys():
                    return data
                else:
                    return None

    def _get_file_name(self):
        styles = os.listdir(UTF_STYLE_DIR)
        check = [i for i in styles if self.name in i and 'json' in i]
        if len(check) == 1:
            return check[0]
        else:
            index = [i for i in check if i.split(".")[0] == self.name]
            if len(index) == 1:
                return index[0]
            else:
                return None

    def _get_utf_style_function(self, data):
        if "function" in data.keys():
            funct = data["function"]
            if "function_arguments" in data.keys():
                funct_arg = data["function_arguments"]
            else:
                funct_arg = None
        else:
            funct = None
            funct_arg = None
        return funct, funct_arg


def get_global_style(style_name):
    stl = MplFigureStyle(style_name)
    style = stl.get_style()
    if style is None:
        stl = UtfFigureStyle(style_name)
        style = stl.get_style()
    if style is None:
        raise Exception('Not a valid matplotlib or utf styles')
    else:
        return style[0], style[1], style[2]


def use_style(func):
    """
    This decorator change the plot style momentaneosly if it is use infront
    of a plotting function and this has the key style
    """
    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    # @wraps use to keep meta data of func
    @wraps(func)
    def style_func(*args, **kwargs):
        valores = dict(zip(argnames, args), **kwargs)
        if func.__defaults__ is not None:
            defaults = dict(zip(argnames[-len(func.__defaults__):],
                                func.__defaults__))
            for i in defaults.keys():
                if i not in valores.keys():
                    valores[i] = defaults[i]
        style, func_plot, func_plot_arg, res = None, None, None, None
        if "style" in valores.keys():
            style = valores["style"]
            try:
                style, func_plot, func_plot_arg = get_global_style(style)
                if "style" in kwargs.keys():
                    kwargs.pop("style")
                with plt.style.context(style):
                    res = func(*args, **kwargs)
                    print('style applied')
            except Exception as m:
                print(m)
                print(1)
                print('style not applied')
                res = func(*args, **kwargs)
            finally:
                if func_plot is not None:
                    try:
                        real_func_plot = globals().get(func_plot)
                        if func_plot_arg is not None:
                            real_func_plot(func_plot_arg)
                        else:
                            real_func_plot()
                    except:
                        print('style function not applied')
                return res
        else:
            return func(*args, **kwargs)
    return style_func
