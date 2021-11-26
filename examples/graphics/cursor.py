# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:31:09 2021

@author: lucas
"""
from ultrafast.graphics.cursors import SnappedCursor
import matplotlib.pyplot as plt
import numpy as np

#generate data for ploting
x = np.linspace(1, 10, 101)
y = np.sin(x)

fig, ax = plt.subplots(1, figsize=(18, 6))
ax.plot(x, y)
cursor = SnappedCursor(ax, x, y)

plt.show()