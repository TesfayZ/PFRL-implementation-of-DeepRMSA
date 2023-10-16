import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
'''
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
'''
import stable_baselines3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter

log_dir = "./tmp/deeprmsa-ppo/"
os.makedirs(log_dir, exist_ok=True)

#results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "DeepRMSA PPO in SB3")
# first, we need to load the monitor data
training_data = pd.read_csv(log_dir + 'training.monitor.csv', skiprows=1)
training_data.describe()
plotting_average_window = 100

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))

ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')

ax2.semilogy(np.convolve(training_data['episode_service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode service blocking rate')

ax3.semilogy(np.convolve(training_data['episode_bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

ax3.set_xlabel('Episode')
ax3.set_ylabel('Episode bit rate blocking rate')
fig.suptitle("DeepRMSA PPO in SB3", fontsize=16)
# fig.get_size_inches()
plt.tight_layout()
plt.show()