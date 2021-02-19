#!/usr/bin/env python3

from ultrafast.tools import readData, select_traces
from ultrafast.experimentClass import Experiment
import matplotlib.pyplot as plt

path = 'examples/data/denoised_2.csv'
time, data, wave = readData(path, wave_is_row=True)

exp = Experiment(time, data, wave)

exp.plotSVD()

plt.show()
