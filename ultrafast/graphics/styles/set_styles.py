"""
Created on Thu Feb 25 14:31:39 2021

@author: lucas
"""
from pathlib import Path
import re
import os
from os.path import join, dirname, realpath
import matplotlib.pyplot as plt
import json
from functools import wraps
from  ultrafast.graphics.styles.plot_base_functions import *

UTF_STYLE_DIR = dirname(__file__)
UTF_BASIC_STYLES = realpath(join(UTF_STYLE_DIR, 'basic_styles'))
STYLE_EXTENSION = 'mplstyle'
STYLE_FILE_PATTERN = re.compile(r'([\S]+).%s$' % STYLE_EXTENSION)

base = plt.style.core.load_base_library()
base_update = plt.style.core.update_user_library(base)
utf = plt.style.core.read_style_directory(UTF_BASIC_STYLES)

library = {**base_update, **utf}


def check_utf_style(name: str):
    styles = os.listdir(UTF_STYLE_DIR)
    check = [i for i in styles if name in i and 'json' in i]
    if len(check) == 1:
         return check[0] 
    else:
        index = [i for i in check if i.split(".")[0] == name]
        if len(index) == 1:
            return index[0] 
        else:
            return None


def is_utf_style(name: str):
    val = check_utf_style(name)
    if val is not None:
        path = realpath(join(UTF_STYLE_DIR, val))
        with open(path, 'r') as f:
            data = json.load(f)
            if "utf_style" in data.keys():
                return data
            else:
                False


def check_if_valid_style(name: str):
    return True if name in library.keys() else False


def get_utf_style(name: str):
    data = is_utf_style(name)
    if data is not False:
        styles = data["styles"]
        style = get_combined_style(styles)
        funct, funct_arg = _get_utf_style_function(data)
        return style, funct, funct_arg
    else:
        msg = 'Not a valid ultrafast style'
        raise Exception(msg)


def _get_utf_style_function(data):
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


def get_combined_style(styles):
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


def get_global_style(style_name):
    style = is_utf_style(style_name)
    if style is None:
        check = check_if_valid_style(style_name)
        if check:
            return library[style_name], None, None
        else:
            msg = 'Not a valid ultrafast style'
            raise Exception(msg)
    else:
        style, func, func_arg = get_utf_style(style_name)
        return style, func, func_arg



def use_style(func):
    """
    This decorator change the plot style momentaneosly if it is use infront
    of a plotting function and this has the key style
    """
    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    if len(argnames) > 0:
        if argnames[0] == 'self':
            argnames = argnames[1:]

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
        print(valores)
        if "style" in valores.keys():
            try:
                style = valores["style"]
                style, func_plot, func_plot_arg = get_global_style(style)
                if "style" in kwargs.keys():
                    kwargs.pop("style")
                with plt.style.context(style):
                    print('style applied')
                    res = func(*args, **kwargs)
            except Exception:
                res = None
                print('style not applied')
                return func(*args, **kwargs)
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
