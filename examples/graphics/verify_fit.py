#!/usr/bin/env python3

from ultrafast.tools import readData, select_traces
from ultrafast.experimentClass import Experiment
import matplotlib.pyplot as plt

path = 'examples/data/denoised_2.csv'
time, data, wave = readData(path, wave_is_row=True)

exp = Experiment(time, data, wave)

exp.select_traces()
exp.initialize_exp_params(0, None, 4, 40, 500)
exp.final_fit()
exp.verify_fit()

plt.show()
