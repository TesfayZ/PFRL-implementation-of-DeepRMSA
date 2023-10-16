# PFRL-implementation-of-DeepRMSA
This is PFRL (Pytorch version of ChainerRL) implementation of [DeepPRMSA] (https://ieeexplore.ieee.org/document/8738827).
I changed the stablebaseline implementation shown in the following link to PFRL. https://github.com/carlosnatalino/optical-rl-gym/blob/main/examples/stable_baselines3/DeepRMSA.ipynb
To run this, first install the optical-rl-gym package. Then, copy the files in the PFRL directory to the directory of the optical-rl-gym package.
Create a directory tmp/deeprmsa-ppo to save the checkpoints and results.
Based on your machine and time requirements, you can reduce the number of steps and evaluations
