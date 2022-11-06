# Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning
## Introduction
- This repository is the implementation of "Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning".
- This study proposed a dual (time and spcace)-domain energy harvesting (EH) approach to maximize EH efficiency of the UAV—RIS system by jointly optimizing the RIS phase shifts vector, the RIS scheduling martix, the length of energy harvesting phase, and the transmit power. 
- For the UAV trajectory design, we considered the density-aware and Fermat point-based algorithms.
- The implementation of DDPG and TD3 using [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/).
- The implementation of SD3 is based on Dr. Pan's research: [Softmax Deep Double Deterministic Policy Gradients](https://github.com/ling-pan/SD3).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me (peng.ee07@nycu.edu.tw). Your suggestions are greatly appreciated.

## Citing
Please consider **citing** our paper if this repository is helpful to you.
**Bibtex:**
```
Haoran Peng, and Li-Chun Wang, “Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning”, has been submitted to IEEE Trans. Wireless Commun. 
```
## Requirements
- Python: 3.6.13
- Pytorch: 1.10.1
- gym: 0.15.3
- numpy: 1.19.2
- matplotlib
- pandas
- Stable-Baselines3

## Usage
#### Descriptions of folders
- The folder "XXXX-MultiUT-Time" is the source code for the time-domain EH scheme in the multiple user scenario.
- The folder "XXXX-MultiUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the multiple user scenario.
- The folder "XXXX-SingleUT-Time" is the source code for the time-domain EH scheme in the single user scenario.
- The folder "XXXX-SingleUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the single user scenario.
- The folder "CreateData" is the source code for generating dataset of trajectories files for users and the UAV.

#### Descriptions of files
- For the Exhaustive Algorithm, the communication environment is impletemented in 'ARIS_ENV.py'.
- For DRL-based algorithms, the communication environment is impletemented in 'gym_foo/envs/foo_env.py'.
- You can change the dataset and the scenario in 'gym_foo/envs/foo_env.py'.

#### Training phase
1. For the TD3 and DDPG, please execute the TD3.py and DDPG.py to train the model, such as
```
python TD3.py / python DDPG.py
```
2. For the exhaustive search, please execute the ExhaustiveSearch.py to reproduce the simulation results.
3. For the SD3, please execute main.py to train a new model. 

#### Testing phase
Please execute test.py to evaluate DRL models. Before you produce the testing results, please change the dataset and scenario in 'gym_foo/envs/foo_env.py'.

#### The EH efficiency
The EH efficiency = the harvested energy / the received energy from RF signals
