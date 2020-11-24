# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:11:02 2020

@author: lucas
"""

class ExperimentException(Exception):
    """General Purpose Exception."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return "{}".format(self.msg)
    
class DataSetContainer:

    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)