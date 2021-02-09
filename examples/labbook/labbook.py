# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:06:54 2021

@author: lucas
"""
from numpy import mean
from ultrafast.tools import LabBook, book_annotate, UnvariableContainer

# The lab book need to be created first
book = LabBook(notes=None, name='Lab Book 1')

def dump_args(func):
     """
     This decorator dumps out the arguments passed to a function before calling it
     """
     @wraps(func)
     def echo_func(*args, **kwargs):
         argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
         valores = dict(zip(argnames, args), **kwargs)
         defaults = dict(zip(argnames[-len(func.__defaults__):] ,func.__defaults__))  
         for i in defaults.keys():
             if i not in valores.keys():
                 valores[i] = defaults[i]
         print(valores)
         return func(*args, **kwargs)
     return echo_func

@book_annotate(book, True)
def square(x):
    return x**2


@book_annotate(book, False)
def mul_3(x):
    return x*3


@book_annotate(book, True)
@dump_args
def suma(a, b=5):
    return a + b


@book_annotate(book, True)
def sum_list(list):
    return sum(list)


# add a function after it has been defined
# step 1: define the function
def resta(a, b):
    return a-b


# step 1: redefine the function with decorator
resta = book_annotate(book, True)(resta)


square(4)
square(5)
mul_3(25)
mul_3(14)
suma(4)
suma(3, 10)
suma(5, 11)
sum_list([14, 8])
resta(14, 8)

book.print()

# printing out:
    
# Lab Book 1
# ----------
# 	 square:
# 		 x = 4
# 		 x = 5


# 	 mul_3:
# 		 x = 14


# 	 suma:
# 		 a = 4, b = 5
# 		 a = 3, b = 10
# 		 a = 5, b = 11


# 	 sum_list:
# 		 list = [14, 8]


# 	 resta:
# 		 a = 14, b = 8


# 	 notes:
# 		 None


# 	 creation:
# 		 day: 09 Jan 2021 | hour: 20:07:14
