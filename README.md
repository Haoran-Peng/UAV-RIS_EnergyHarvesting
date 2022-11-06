# Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning
## Introduction
- This repository is the implementation of "Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning".
- This study proposed a dual (time and spcace)-domain energy harvesting (EH) approach to maximize EH efficiency of the UAV—RIS system by jointly optimizing the RIS phase shifts vector, the RIS scheduling martix, the length of energy harvesting phase, and the transmit power. 
- For the UAV trajectory design, we considered the density-aware and Fermat point-based algorithms.
- The implementation of DDPG and TD3 using [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/).
- The implementation of SD3 is based on Dr. Pan's research: [Softmax Deep Double Deterministic Policy Gradients](https://github.com/ling-pan/SD3).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me. Your suggestions are greatly appreciated.

## Citing
Please consider **citing** our paper if this repository is helpful to you.
**Bibtex:**
```
@INPROCEEDINGS{peng1570767WCNC,
  author={Peng, Haoran and Wang, Li-Chun},
  booktitle={Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)}, 
  title={Long-Lasting {UAV}-aided {RIS} Communications based on {SWIPT}},
  address={Austin, TX},
  year={2022},
  month = {Apr.}
}
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
1. In the main.py, the switch of "Train" must be 'True' such as
```
14 Train = True # True for tranining, and False for testing.
```
2. python main.py

#### Testing phase
1. In the main.py, the switch of "Train" must be 'False' such as
```
14 Train = False # True for tranining, and False for testing.
```
2. python main.py
3. The harvest energy of each step and overall steps are saved in 'Test_Rewards_Records.csv' and 'Total_Reward_Test.txt', respectively.

#### The EH efficiency
The EH efficiency for each step can be calculated by:
```
 reward of each step / 0.02275827153828275
```
