#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import math
################################
#excuse this file will generate 20 distance dataset.
################################

################################
#       20m  20m
#      ____ ____ 
#     |    |    | 20m
#     |____|____|
#     |    |    | 20m
#     |____|____|
################################
class UT:
	CurX = 0
	CurY = 0
	CurZ = 0
	TargetX = 0
	TargetY = 0
	Trajectory = []
	State = False # True means this UE has arrived the destination

	def __init__(self, InitX, InitY, TargetX, TargetY):
		self.CurX = InitX
		self.CurY = InitY
		self.TargetX = TargetX
		self.TargetY = TargetY
		self.Trajectory = []

	def saveTrajectory(self):
		self.Trajectory.append([self.CurX, self.CurY, 1])

	def move(self):
		if abs(self.TargetX-self.CurX)>=abs(self.TargetY-self.CurY):
			moveX = lambda Cx, Tx: (Cx - 1) if Cx > Tx else Cx + 1
			self.CurX = moveX(self.CurX, self.TargetX)
		else:
			moveY = lambda Cy, Ty: (Cy - 1) if Cy > Ty else Cy + 1
			self.CurY = moveY(self.CurY, self.TargetY)

		if (self.TargetX==self.CurX)&(self.TargetY==self.CurY):
			self.State = True

def TrainDate(num):
	DistanceSet = []
	Location = []
	if num == 1:
		Location = [[0, 0, 20, 20]]
	else:
		Location = [[0, 0, 20, 20], [20, 0, 0, 20], [20, 20, 0, 0]]

	for i in range(len(Location)):
		Agent = UT(Location[i][0], Location[i][1], Location[i][2], Location[i][3])
		Agent.saveTrajectory()
		while True:
			Agent.move()
			Agent.saveTrajectory()
			if Agent.State == True:
				break

		Traj_data = np.array(Agent.Trajectory)
		Traj_data.reshape(3, len(Traj_data))
		np.savetxt("Train_Trajectory_UT_"+str(i)+".csv", Traj_data, delimiter=',')	

def TestData(num):
	DistanceSet = []
	Location = []
	if num == 1:
		Location = [[20, 0, 0, 20]]
	else:
		Location = [[20, 20, 0, 0], [0, 20, 20, 0], [0, 0, 20, 20]]

	for i in range(len(Location)):
		Agent = UT(Location[i][0], Location[i][1], Location[i][2], Location[i][3])
		Agent.saveTrajectory()
		while True:
			Agent.move()
			Agent.saveTrajectory()
			if Agent.State == True:
				break
		Traj_data = np.array(Agent.Trajectory)
		Traj_data.reshape(3, len(Traj_data))
		np.savetxt("Test_Trajectory_UT_"+str(i)+".csv", Traj_data, delimiter=',')

if __name__ == '__main__':
	##Single UT case
	TrainDate(1)
	TestData(1)

	# ##Multiple UT case
	TrainDate(3)
	TestData(3)