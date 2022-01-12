#!/usr/bin/env python3

from ultrapyfit.utils.divers import read_data, select_traces
from ultrapyfit.experiment import Experiment
import matplotlib.pyplot as plt

path = 'examples/data/denoised_2.csv'
time, data, wave = read_data(path, wave_is_row=True)

exp = Experiment(time, data, wave)

exp.plot_full_SVD()

plt.show()
