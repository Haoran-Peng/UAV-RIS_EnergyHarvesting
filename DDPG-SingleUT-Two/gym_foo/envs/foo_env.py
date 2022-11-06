import gym
from gym import error, spaces, utils
from gym.utils import seeding
import globe
import numpy as np
import random as rd
import time
import math as mt
import sys
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, LoadData = True, Train = False, multiUT = False, Trajectory_mode = 'Kmeans', MaxStep = 41):
        globe._init()
        #the initial location of UAV-RIS
        globe.set_value('L_U', [0, 0, 20]) #[x, y, z]
        #the location of AP/BS
        globe.set_value('L_AP', [0, 0, 10])
        #8 antennas for AP
        globe.set_value('BS_Z', 8)
        #16 reflective elements for RIS
        globe.set_value('RIS_L', 16)

        #CSI parameters
        globe.set_value('BW', 2 * mt.pow(10, 7)) #The bandwidth is 20 MHz
        # Noise power spectrum density is -174dBm/Hz;
        globe.set_value('N_0', mt.pow(10, ((-174 / 3) / 10)))
        globe.set_value('Xi', mt.pow(10, (3/10))) #the path loss at the reference distance D0 = 1m, 3dB;
        # urban env. from [Efficient 3-D Placement of an Aerial Base Station in Next Generation Cellular Networks] 
        #and [Joint Trajectory-Task-Cache Optimization with Phase-Shift Design of RIS-Assisted UAV for MEC]
        globe.set_value('a', 9.61)
        globe.set_value('b', 0.16)
        globe.set_value('eta_los', 1) 
        globe.set_value('eta_nlos', 20)

        # σ2 = -102dBm
        globe.set_value('AWGN', mt.pow(10, (-102/10)))
        # number of RIS antenna
        globe.set_value('N_ris', 100)
        #energy harvesting efficiency eta=0.7
        globe.set_value('eta', 0.7) 
        #path-loss exponent is α=3
        globe.set_value('alpha', 3)
        # additional attenuation factor φ is 20 dB
        globe.set_value('varphi', mt.pow(10, (20/10)))
        # max transmit Power from BS is 500W
        globe.set_value('P_max', 5 * mt.pow(10, 5))#mt.pow(10, 43/10)
        #number of user 
        globe.set_value('N_u', 3)
        # carrier frequency is 750 MHz
        globe.set_value('fc', 750 * mt.pow(10, 6))
        # speed of light
        globe.set_value('c', 3 * mt.pow(10, 8))
        #minimal requirement of sinr 12db
        globe.set_value('gamma_min', mt.pow(10, (12/10)))
        # the transmission power from the AP for a single user
        globe.set_value('power_i', 0.5 * mt.pow(10, 3))
        # the number of total time slots
        globe.set_value('t', int(MaxStep))
        #current time slot
        globe.set_value('step', 0)

        globe.set_value('kappa', mt.pow(10, (-30/10)))
        globe.set_value('hat_alpha', 2)
        globe.set_value('successCon', 0)

        if LoadData == True:
            if Train == True:
                if multiUT == True:
                    UT_0 = np.loadtxt("../CreateData/Train_Trajectory_UT_0.csv", delimiter=",")
                    UT_1 = np.loadtxt("../CreateData/Train_Trajectory_UT_1.csv", delimiter=",")
                    UT_2 = np.loadtxt("../CreateData/Train_Trajectory_UT_2.csv", delimiter=",")

                    globe.set_value('UT_0', UT_0)
                    globe.set_value('UT_1', UT_1)
                    globe.set_value('UT_2', UT_2)

                    if Trajectory_mode == 'Fermat':
                        UAV_Trajectory = np.loadtxt("../CreateData/Fermat_Train_Trajectory_3.csv", delimiter=",")
                    else:
                        UAV_Trajectory = np.loadtxt("../CreateData/Kmeans_Train_Trajectory_3.csv", delimiter=",")

                    globe.set_value('UAV_Trajectory', UAV_Trajectory)

                else:
                    UT_0 = np.loadtxt("../CreateData/Train_Trajectory_UT_0.csv", delimiter=",")
                    globe.set_value('UT_0', UT_0)

                    if Trajectory_mode == 'Fermat':
                        UAV_Trajectory = np.loadtxt("../CreateData/Fermat_Train_Trajectory_1.csv", delimiter=",")
                    else:
                        UAV_Trajectory = np.loadtxt("../CreateData/Kmeans_Train_Trajectory_1.csv", delimiter=",")

                    globe.set_value('UAV_Trajectory', UAV_Trajectory)

            else:
                if multiUT == True:
                    UT_0 = np.loadtxt("../CreateData/Test_Trajectory_UT_0.csv", delimiter=",")
                    UT_1 = np.loadtxt("../CreateData/Test_Trajectory_UT_1.csv", delimiter=",")
                    UT_2 = np.loadtxt("../CreateData/Test_Trajectory_UT_2.csv", delimiter=",")

                    globe.set_value('UT_0', UT_0)
                    globe.set_value('UT_1', UT_1)
                    globe.set_value('UT_2', UT_2)

                    if Trajectory_mode == 'Fermat':
                        UAV_Trajectory = np.loadtxt("../CreateData/Fermat_Test_Trajectory_3.csv", delimiter=",")
                    else:
                        UAV_Trajectory = np.loadtxt("../CreateData/Kmeans_Test_Trajectory_3.csv", delimiter=",")

                    globe.set_value('UAV_Trajectory', UAV_Trajectory)

                else:
                    UT_0 = np.loadtxt("../CreateData/Test_Trajectory_UT_0.csv", delimiter=",")
                    globe.set_value('UT_0', UT_0)

                    if Trajectory_mode == 'Fermat':
                        UAV_Trajectory = np.loadtxt("../CreateData/Fermat_Test_Trajectory_1.csv", delimiter=",")
                    else:
                        UAV_Trajectory = np.loadtxt("../CreateData/Kmeans_Test_Trajectory_1.csv", delimiter=",")

                    globe.set_value('UAV_Trajectory', UAV_Trajectory)        
        
        self.action_space = spaces.Box(0, 1, shape=(34, ), dtype=np.float32)
        self.observation_space = spaces.Box(0, 20, shape=(2, ), dtype=np.float32)
        self._max_episode_steps = 41
        self.Train = Train

    def step(self, action):
        t = globe.get_value('t')
        tau = action[0] # The length of the EH phase
        power_1 = mt.pow(10, ((action[1]-1)*30/10+3)) # power for UT 1
        Theta_R = action[2: 2 + globe.get_value('RIS_L')] * 2 * np.pi
        Omega_R = np.abs(np.around(action[2 + globe.get_value('RIS_L'):]))

        step = globe.get_value('step')
        reward, radio_state, received_energy = self.env_state(step, tau, power_1, Theta_R, Omega_R)
        done = False
        if step == t - 1:
            done = True
   
        globe.set_value('step', int(step+1))

        radio_state = radio_state/np.sum(radio_state)

        if (np.array(action) >= 0).all() == False:
            reward = 0

        if (np.array(action) <= 1).all() == False:
            reward = 0

        if self.Train == True:
            return radio_state, reward/received_energy, done, {}
        else:
            return radio_state, reward/received_energy, done, {str(reward)+","+str(received_energy)} 

    def reset(self):
        globe.set_value('step', 0)
        globe.set_value('successCon', 0)
        
        step = globe.get_value('step')

        L_U = globe.get_value('UAV_Trajectory')[step]
        L_AP = globe.get_value('L_AP')

        UT_0 = globe.get_value('UT_0')[step]

        distance_AP_RIS = mt.sqrt(mt.pow((L_U[0] - L_AP[0]), 2) + mt.pow((L_U[1] - L_AP[1]), 2) + mt.pow((L_U[2] - L_AP[2]), 2))
        distance_RIS_UT_0 = mt.sqrt(mt.pow((L_U[0] - UT_0[0]), 2) + mt.pow((L_U[1] - UT_0[1]), 2) + mt.pow((L_U[2] - UT_0[2]), 2))

        radio_state = np.array([distance_AP_RIS, distance_RIS_UT_0])
        radio_state = radio_state/np.sum(radio_state)
        return radio_state    

    def render(self, mode='human', close=False):
        pass

    def pl_BR(self, L_U, L_AP):
        a = globe.get_value('a')
        b = globe.get_value('b')
        varphi = globe.get_value('varphi')
        alpha = globe.get_value('alpha')

        theta = (180 / mt.pi) * mt.asin( ( (L_U[2] - L_AP[2]) / mt.sqrt(mt.pow(L_U[0], 2) + mt.pow(L_U[1], 2) + mt.pow((L_U[2] - L_AP[2]), 2))) )
        p_los = 1 + a * mt.exp(a * b - b * theta )
        p_los = 1 / p_los

        p_nlos = 1 - p_los
        # channel power gain (BS-RIS) with the los and nlos
        g_BR = (p_los + p_nlos * varphi) * mt.pow(mt.sqrt(mt.pow(L_U[0], 2) + mt.pow(L_U[1], 2) + mt.pow((L_U[2] - L_AP[2]), 2)), (0-alpha))
        
        return g_BR

    def SmallFading_G(self, BS_Z, RIS_L):
        SmallFading_G = 1/np.sqrt(2)*(np.random.normal(loc=0, scale=1, size=(BS_Z, RIS_L)) + 1j*np.random.normal(loc=0, scale=1, size=(BS_Z, RIS_L)))
        return SmallFading_G

    def EH(self, tau, power_1, Theta_R, L_U, L_AP, Omega_R):
        eta = globe.get_value('eta')
        #8 antennas for AP
        BS_Z = globe.get_value('BS_Z')
        #16 reflective elements for RIS
        RIS_L = globe.get_value('RIS_L')

        g_BR = self.pl_BR(L_U, L_AP)
        # 8 antennas for AP, 16 reflective elements for RIS
        G = np.ones((BS_Z, RIS_L))
        SmallFading_G = self.SmallFading_G(BS_Z, RIS_L)
        x = G * g_BR * SmallFading_G

        power_total = power_1
        g_2norm = np.zeros((BS_Z, RIS_L))

        for i in range(0, BS_Z):
            for j in range(0, RIS_L):
                g_ij = [x[i][j]]
                g_2norm[i][j] = np.linalg.norm(g_ij, ord=2, keepdims=True)

        received_power = np.sum(g_2norm * power_total)

        g_hat_2norm = np.zeros((BS_Z, RIS_L))
        for i in range(0, BS_Z):
            for j in range(0, RIS_L):
                g_ij = [x[i][j]]
                g_hat_2norm[i][j] = np.linalg.norm(g_ij, ord=2, keepdims=True) * (1 - Omega_R[j])

        received_power_hat = np.sum(g_hat_2norm * power_total)

        E_t = tau * eta * received_power + (1 - tau) * eta * received_power_hat

        return E_t, received_power

    def Rayleigh_RU(self, L_U, UT, BS_Z, RIS_L):
        Rayleigh_RU = 1/np.sqrt(2)*(np.random.normal(loc=0, scale=1, size=(RIS_L, 1)) + 1j*np.random.normal(loc=0, scale=1, size=(RIS_L, 1)))
        return Rayleigh_RU
    
    def Channel_RU(self, L_U, UT, BS_Z, RIS_L):
        h_ru = np.ones((RIS_L, 1))
        kappa = globe.get_value('kappa')
        hat_alpha = globe.get_value('hat_alpha')
        distance = mt.sqrt(mt.pow((L_U[0] - UT[0]), 2) + mt.pow((L_U[1] - UT[1]), 2) + mt.pow((L_U[2] - UT[2]), 2))
        PL = np.sqrt(kappa * mt.pow((distance/1), -hat_alpha))

        h_ru = h_ru * np.sqrt(5/(1+5)) * PL + np.sqrt(1/(1+5)) * PL * self.Rayleigh_RU(L_U, UT, BS_Z, RIS_L)

        return h_ru

    def capacity (self, tau, power_1, Theta_R, L_U, L_AP, UT_0, Omega_R):

        power_i = globe.get_value('power_i')
        AWGN = globe.get_value('AWGN')

        BW = globe.get_value('BW')
        # 8 antennas for AP
        BS_Z = globe.get_value('BS_Z')
        # 16 reflective elements for RIS
        RIS_L = globe.get_value('RIS_L')

        g_BR = self.pl_BR(L_U, L_AP)
        SmallFading_G = self.SmallFading_G(BS_Z, RIS_L)
        G = np.ones((BS_Z, RIS_L)) * g_BR * SmallFading_G

        coefficients = np.diag(np.exp(1j*Theta_R))

        for i in range(RIS_L):
            coefficients[i][i] = coefficients[i][i] * Omega_R[i]

        # received signal for UT 1
        h_ru = self.Channel_RU(L_U, UT_0, BS_Z, RIS_L)    
        UT_link_1 = 1*np.linalg.multi_dot([G, coefficients, h_ru])
        signal_UT_1 = np.sum(np.abs(UT_link_1 * np.conjugate(UT_link_1))) * power_1

        SINR_1 = 10 * mt.log(((signal_UT_1 + 1e-14)/(AWGN)), 10)

        if SINR_1 > 0:
            Aver_Throughput_1 = BW * mt.log((1 + SINR_1), 2) * (1 - tau)
        else:
            Aver_Throughput_1 = 0

        return [Aver_Throughput_1]

    def env_state(self, step, tau, power_1, Theta_R, Omega_R):
        if step < globe.get_value('t')-1:
            L_U = globe.get_value('UAV_Trajectory')[step+1]
            L_AP = globe.get_value('L_AP')

            UT_0 = globe.get_value('UT_0')[step+1]
        else:
            L_U = globe.get_value('UAV_Trajectory')[step]
            L_AP = globe.get_value('L_AP')

            UT_0 = globe.get_value('UT_0')[step]

        reward, received_energy = self.EH(tau, power_1, Theta_R, L_U, L_AP, Omega_R)
        HarvestEnergy = reward
        Aver_Throughput = self.capacity (tau, power_1, Theta_R, L_U, L_AP, UT_0, Omega_R)

        for i in range(len(Aver_Throughput)):
            if Aver_Throughput[i] < 7 * mt.pow(10, 7):
                reward = 0
                HarvestEnergy = 0

        distance_AP_RIS = mt.sqrt(mt.pow((L_U[0] - L_AP[0]), 2) + mt.pow((L_U[1] - L_AP[1]), 2) + mt.pow((L_U[2] - L_AP[2]), 2))
        distance_RIS_UT_0 = mt.sqrt(mt.pow((L_U[0] - UT_0[0]), 2) + mt.pow((L_U[1] - UT_0[1]), 2) + mt.pow((L_U[2] - UT_0[2]), 2))

        radio_state = np.array([distance_AP_RIS, distance_RIS_UT_0])

        return reward, radio_state, received_energy

    def reloadData(filename):
        with open(filename, encoding = 'utf-8') as f:
            data = np.loadtxt(f, delimiter = ",")
            data.astype(np.int)
            globe.set_value('DistanceRU', data)
