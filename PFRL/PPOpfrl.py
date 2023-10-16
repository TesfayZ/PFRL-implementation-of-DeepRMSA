import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from customSBmonitor import Monitor
import logging

import pfrl
from pfrl import experiments
from pfrl.agents import PPO
from pfrl.experiments import EvaluationHook
from pfrl import agents, experiments

import gym

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

log_dir = "./tmp/deeprmsa-ppo/"
os.makedirs(log_dir, exist_ok=True)
# Load the topology from the pickle file
topology_name = 'nsfnet_chen'
k_paths = 5
with open(f'./examples/topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
    topology = pickle.load(f)

monitor_info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate')

# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])

# mean_service_holding_time=7.5,
env_args = dict(topology=topology, seed=10, 
                allow_rejection=False, # the agent cannot proactively reject a request
                j=1, # consider only the first suitable spectrum block for the spectrum assignment
                mean_service_holding_time=7.5, # value is not set as in the paper to achieve comparable reward values
                episode_length=50, node_request_probabilities=node_request_probabilities)
#print(topology.edges())
# Create the environment
env = gym.make('DeepRMSA-v0', **env_args)
'''
env = PFRlMonitor(
    env=env,
    directory=log_dir +'custom_logs/',
    video_callable=None,
    force=True,
    resume=False,
    write_upon_reset=True,
    uid="custom_monitor",
    mode="training",
    info_keys=["episode_service_blocking_rate", "episode_bit_rate_blocking_rate"],
)
'''
env = Monitor(env, log_dir + 'PFRLtraining', info_keywords=monitor_info_keywords)
#env = Monitor(env, log_dir + 'training', force=True)
#env = Monitor(env, log_dir, force=True)
obs_size = env.observation_space.shape[0]
action_size = env.action_space.n

# policy_args = dict(net_arch=5*[128]) # we use the elu activation function as in https://github.com/carlosnatalino/optical-rl-gym/blob/main/examples/stable_baselines3/DeepRMSA.ipynb
hl=128
# becuase the action must select the best lightpath, the best action selector policy from pfrl is SoftmaxCategoricalHead(), from the choices https://pfrl.readthedocs.io/en/latest/policies.html#pfrl.policies.SoftmaxCategoricalHead
policy = torch.nn.Sequential(
    nn.Linear(obs_size, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, action_size),
    pfrl.policies.SoftmaxCategoricalHead(),
)
vf = torch.nn.Sequential(
    nn.Linear(obs_size, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, hl),
    nn.ELU(),
    nn.Linear(hl, 1),
)


#we used orthon_int because ortho_int = True in, https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html#ActorCriticPolicy, 
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)

ortho_init(policy[0], gain=1)
ortho_init(policy[2], gain=1)
ortho_init(policy[4], gain=1)
ortho_init(policy[6], gain=1)
ortho_init(policy[8], gain=1)
ortho_init(policy[10], gain=1e-2)
ortho_init(vf[0], gain=1)
ortho_init(vf[2], gain=1)
ortho_init(vf[4], gain=1)
ortho_init(vf[6], gain=1)
ortho_init(vf[8], gain=1)
ortho_init(vf[10], gain=1)

# Combine a policy and a value function into a single model
model = pfrl.nn.Branched(policy, vf)

optimizer = Adam(params=model.parameters(), lr=10e-6, eps=1e-5) # based on https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html#ActorCriticPolicy

# parameters are based on https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
ppo_agent = PPO(model=model,
    optimizer=optimizer,
    gpu=-1,
    gamma=0.95,
    lambd=0.95,
    phi=lambda x: x.astype(np.float32, copy=False),
    update_interval=2048,
    minibatch_size=64,
    epochs=10,
    clip_eps=0.2,
    clip_eps_vf=None,
    standardize_advantages=True,
    entropy_coef=0.0,
    value_func_coef=0.5,
    max_grad_norm=0.5,
    ) 

    
total_timesteps = 10000000
check_freq = 1000
experiments.train_agent_with_evaluation(
    agent=ppo_agent,
    env=env,
    eval_env=None,
    eval_n_steps=None,
    outdir=log_dir,
    steps=total_timesteps,
    eval_interval=10000,
    eval_n_episodes=100,
    checkpoint_freq=check_freq,
    step_offset=0,
    save_best_so_far_agent=True,
)
'''
experiments.train_agent(
    agent=ppo_agent,
    env=env,
    outdir=log_dir,
    steps=total_timesteps,
    checkpoint_freq=check_freq,
    step_offset=0,

)
'''
'''
successful_score=None,????
logger=logger, # inlcude this inside experiments.train_agent_with_evaluation to see progress of training in console
logger=lambda x: custom_logger(env, ppo_agent, x, log_dir +'custom_logs.csv') #use this to log information to file
'''


training_data = pd.read_csv(log_dir + 'PFRLtraining.monitor.csv', skiprows=1)
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

# fig.get_size_inches()
plt.tight_layout()
plt.show()


