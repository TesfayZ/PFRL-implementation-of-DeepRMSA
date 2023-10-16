# PFRL-implementation-of-DeepRMSA
This is a PFRL (Pytorch version of ChainerRL) implementation of [DeepPRMSA] (https://ieeexplore.ieee.org/document/8738827)

Chainerrl and stable baseline are reinforcement learning libraries. The Chainerrl library is tensorflow implementation.  PFRL is the pytorch version of Chaninerrl. 

The stable baseline implementation of DeepRMSA by Carlos Natalino is here: https://github.com/carlosnatalino/optical-rl-gym/blob/main/examples/stable_baselines3/DeepRMSA.ipynb

Here, I implemented it in pfrl and found that pfrl provides faster convergence than SB3. 

To run the PFRL implementation, first install the optical-rl-gym package.
Then move the files in PFRL (not SB3) to the directory of the optical-rl-gym package.
create tmp/deeprmsa-ppo to save results and checkpoints

Othe dependencies include the packages of the stable baseline implementation and installation of the pfrl. 

The number of steps and evaluation intervals can be minimized to reduce space ( in saving checkpints) and time. 
