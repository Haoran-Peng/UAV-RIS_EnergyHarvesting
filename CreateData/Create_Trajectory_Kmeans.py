import numpy as np
import matplotlib.pyplot as plt
import math
import random

class kmeans_UAV(object):
    """docstring for kmeans_UAV"""
    def __init__(self):
        super(kmeans_UAV, self).__init__()
        
        self.cluster_num = 1 # The number of cluster
        self.UT_num = 3  # The number of UTs

        # The initial center point
        self.kx = np.random.randint(0, 40, self.cluster_num)
        self.ky = np.random.randint(0, 40, self.cluster_num)

        #The value of center point fot saving
        self.centerX = 0
        self.centerY = 0

    # distance
    def dis(self, x, y, kx, ky):
        return int(((kx-x)**2 + (ky-y)**2)**0.5)

    # cluster
    def cluster(self, x, y, kx, ky):
        team = []
        for i in range(self.cluster_num):
            team.append([])
        mid_dis = 99999999
        for i in range(self.UT_num):
            for j in range(self.cluster_num):
                distant = self.dis(x[i], y[i], kx[j], ky[j])
                if distant < mid_dis:
                    mid_dis = distant
                    flag = j
            team[flag].append([x[i], y[i]])
            mid_dis = 99999999
        return team

    def re_seed(self, team, kx, ky):
        sumx = 0
        sumy = 0
        new_seed = []
        for index, nodes in enumerate(team):
            if nodes == []:
                new_seed.append([kx[index], ky[index]])
            for node in nodes:
                sumx += node[0]
                sumy += node[1]
            new_seed.append([int(sumx/len(nodes)), int(sumy/len(nodes))])
            sumx = 0
            sumy = 0
        nkx = []
        nky = []
        for i in new_seed:
            nkx.append(i[0])
            nky.append(i[1])
        return nkx, nky

    # k-means cluster
    def kmeans(self, x, y, kx, ky):
        team = self.cluster(x, y, kx, ky)
        nkx, nky = self.re_seed(team, kx, ky)
        
        if nkx == list(kx) and nky == (ky):
            self.centerX = kx
            self.centerY = ky
            return
        else:
            self.kmeans(x, y, nkx, nky)

    def Trajectory(self, num_UT, mode, UT_0, UT_1, UT_2):
        UAV_Trajectory = []
        if num_UT == 3:
            for i in range(len(UT_0)):
                x = [UT_0[i][0], UT_1[i][0], UT_2[i][0]]
                y = [UT_0[i][1], UT_1[i][1], UT_2[i][1]]
                self.kmeans(x, y, self.kx, self.ky)
                UAV_Trajectory.append([self.centerX[0], self.centerY[0], 20])

        if num_UT == 1:
            for i in range(len(UT_0)):
                x = UT_0[i][0]
                y = UT_0[i][1]
                UAV_Trajectory.append([x, y, 20])  
        
        Traj_data = np.array(UAV_Trajectory)
        Traj_data.reshape(3, len(UAV_Trajectory))
        np.savetxt("Kmeans_"+str(mode)+"_Trajectory_"+str(num_UT)+".csv", Traj_data, delimiter=',')

    def load_UT_Trajectory(self, mode, num_UT): # mode ==1 is for training data, mode == 2 is for testing data
        if num_UT == 3 and mode == 1:
            UT_0 = np.loadtxt("Train_Trajectory_UT_0.csv", delimiter=",")
            UT_1 = np.loadtxt("Train_Trajectory_UT_1.csv", delimiter=",")
            UT_2 = np.loadtxt("Train_Trajectory_UT_2.csv", delimiter=",")
            self.Trajectory(3, "Train", UT_0, UT_1, UT_2)

        if num_UT == 3 and mode == 2:
            UT_0 = np.loadtxt("Test_Trajectory_UT_0.csv", delimiter=",")
            UT_1 = np.loadtxt("Test_Trajectory_UT_1.csv", delimiter=",")
            UT_2 = np.loadtxt("Test_Trajectory_UT_2.csv", delimiter=",")
            self.Trajectory(3, "Test", UT_0, UT_1, UT_2)

        if num_UT == 1 and mode == 1:
            UT_0 = np.loadtxt("Train_Trajectory_UT_0.csv", delimiter=",")
            self.Trajectory(1, "Train", UT_0, None, None)

        if num_UT == 1 and mode == 2:
            UT_0 = np.loadtxt("Test_Trajectory_UT_0.csv", delimiter=",")
            self.Trajectory(1, "Test", UT_0, None, None)

if __name__ == '__main__':
    kmeans_UAV = kmeans_UAV()
    kmeans_UAV.load_UT_Trajectory(1, 3)
    kmeans_UAV.load_UT_Trajectory(2, 3)
    kmeans_UAV.load_UT_Trajectory(1, 1)
    kmeans_UAV.load_UT_Trajectory(2, 1)