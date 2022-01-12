#!/usr/bin/env python3

from ultrapyfit.utils.divers import read_data, select_traces
from ultrapyfit.experiment import Experiment
import matplotlib.pyplot as plt

path = 'examples/data/denoised_2.csv'
time, data, wave = read_data(path, wave_is_row=True)

exp = Experiment(time, data, wave)

exp.select_traces()
exp.initialize_exp_params(0, None, 4, 40, 500)
exp.global_fit()
exp.verify_fit()

plt.show()