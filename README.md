# Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning
## Introduction
- This repository is the implementation of "Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning".
- This study aims to maximize energy harvesting efficiency of the UAV—RIS system by jointly optimizing the RIS phase shifts vector, the RIS scheduling martix, the length of energy harvesting phase, and the transmit power.
- The implementation of DDPG and TD3 using [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/).
- The implementation of SD3 is based on Dr. Pan's research: [Softmax Deep Double Deterministic Policy Gradients](https://github.com/ling-pan/SD3).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me. Your suggestions are greatly appreciated.


__Our extension of this work maximizes the energy harvesting (EH) efficiency by jointly optimizing the phase shift of the reflecting elements, the transmit power, the length of the EH phase, and the RIS scheduling. Please find the latest repository by the below link:  
[UAV—RIS_EnergyHarvesting for IEEE TWC paper](https://github.com/Haoran-Peng/UAV-RIS_EnergyHarvesting)__

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
- Stable-Baselines3 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

## Usage
#### Descriptions of folders
- The folder "DDPG-SingleUT-Time" is the source code for the time-domain EH scheme using DDPG.
- The folder "DDPG-SingleUT-Time-and-Space" is the source code for the two-domain (Time and Space) EH scheme using DDPG.
- The folder "Exhaustive-SingleUT-Time" is the source code for the time-domain EH scheme using Exhaustive Algorithm.
- The folder "Exhaustive-SingleUT-Time-and-Space" is the source code for the two-domain (Time and Space) EH scheme using  Exhaustive Algorithm.

#### Descriptions of files
- For the Exhaustive Algorithm, the communication environment is impletemented in 'ARIS_ENV.py'.
- For the DDPG, the communication environment is impletemented in 'gym_foo/envs/foo_env.py'.
- You can change the dataset in 'gym_foo/envs/foo_env.py'.

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
