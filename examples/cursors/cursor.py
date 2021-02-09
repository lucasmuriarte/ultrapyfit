#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:31:09 2021

@author: lucas
"""
from ultrafast.MaptplotLibCursor import SnaptoCursor
import matplotlib.pyplot as plt
import numpy as np

#generate data for ploting
x = np.linspace(1,10,101)
y=np.sin(x)

fig,ax=plt.subplots(1,figsize=(18,6))
ax.plot(x,y)
cursor = SnaptoCursor(ax, x,y)
plt.connect('axes_enter_event', cursor.onEnterAxes)
plt.connect('axes_leave_event', cursor.onLeaveAxes)
plt.connect('motion_notify_event', cursor.mouseMove)
plt.connect('button_press_event', cursor.onClick)
fig.show()


cursor.datax #returns the x values after clicking
cursor.datay #returns the y values after clicking
plt.show()