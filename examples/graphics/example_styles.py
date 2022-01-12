# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:31:39 2021

@author: lucas
"""
import numpy as np
import matplotlib.pyplot as plt
from  ultrapyfit.graphics.styles.set_styles import *
from  ultrapyfit.graphics.styles.plot_base_functions import *

# how to use styles:
# first define the function, with the decorator use_style
# call the function with the style you want as attribute

data = np.random.randn(50)
x = np.linspace(1,50, 50)

@use_style
def plot_test():
    plt.figure()
    plt.plot(x, data)
    plt.xlabel('Hola amigos')
    
@use_style
def plot_test2(test=1,style="lmu_trac"):
    print(test)
    plt.figure()
    plt.plot(x, data**2)
    plt.xlabel('Hola amigos')
 
plot_test(style="lmu_trac")    
plot_test2(2)

plot_test(style="lmu_specd")    
plot_test2(style="ggplot")