# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:04:41 2020

@author: 79344
"""
from functools import wraps

def froze_it(cls):
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and hasattr(self, key):
            print("Class {} is frozen. Cannot modified {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls

@froze_it 
class UnvariableContainer:
    
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)

class VariableContainer:
    
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)
            