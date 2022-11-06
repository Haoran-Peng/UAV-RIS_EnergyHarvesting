import numpy as np
import math
import random

def Fermat_point(UT_0, UT_1, UT_2):
	a1 = [UT_0[0], UT_1[0], UT_2[0]]
	a2 = [UT_0[1], UT_1[1], UT_2[1]]

	x=sum(a1)/2
	y=sum(a2)/2

	while True:
		x_nume=0 # numerator
		x_deno=0 # denominator
		y_nume=0
		y_deno=0
		for i in range(2):
			g=math.sqrt((x-a1[i])**2+(y-a2[i])**2)
			if g == 0:
				x_nume = x_nume + a1[i]
				x_deno = x_deno + 1
				y_nume = y_nume + a2[i]
				y_deno = y_deno + 1
			else:
				x_nume = x_nume + a1[i]/g
				x_deno = x_deno + 1/g
				y_nume = y_nume + a2[i]/g
				y_deno = y_deno + 1/g
		xn = x_nume/x_deno
		yn = y_nume/y_deno
		if abs(xn-x)<0.01 and abs(yn-y)<0.01:
			break
		else:
			x=xn
			y=yn

	return x, y

def Fermat_Trajectory(num_UT, mode, UT_0, UT_1, UT_2):
	UAV_Trajectory = []

	if num_UT == 1:
		for i in range(len(UT_0)):
			x = UT_0[i][0]
			y = UT_0[i][1]
			UAV_Trajectory.append([x, y, 20])
	else:
		for i in range(len(UT_0)):
			x, y = Fermat_point(UT_0[i], UT_1[i], UT_2[i])
			UAV_Trajectory.append([x, y, 20])

	Traj_data = np.array(UAV_Trajectory)
	Traj_data.reshape(3, len(UAV_Trajectory))
	np.savetxt("Fermat_"+str(mode)+"_Trajectory_"+str(num_UT)+".csv", Traj_data, delimiter=',')

def load_UT_Trajectory(mode, num_UT): # mode ==1 is for training data, mode == 2 is for testing data
	if num_UT == 3 and mode == 1:
		UT_0 = np.loadtxt("Train_Trajectory_UT_0.csv", delimiter=",")
		UT_1 = np.loadtxt("Train_Trajectory_UT_1.csv", delimiter=",")
		UT_2 = np.loadtxt("Train_Trajectory_UT_2.csv", delimiter=",")
		Fermat_Trajectory(3, "Train", UT_0, UT_1, UT_2)

	if num_UT == 3 and mode == 2:
		UT_0 = np.loadtxt("Test_Trajectory_UT_0.csv", delimiter=",")
		UT_1 = np.loadtxt("Test_Trajectory_UT_1.csv", delimiter=",")
		UT_2 = np.loadtxt("Test_Trajectory_UT_2.csv", delimiter=",")
		Fermat_Trajectory(3, "Test", UT_0, UT_1, UT_2)

	if num_UT == 1 and mode == 1:
		UT_0 = np.loadtxt("Train_Trajectory_UT_0.csv", delimiter=",")
		Fermat_Trajectory(1, "Train", UT_0, None, None)

	if num_UT == 1 and mode == 2:
		UT_0 = np.loadtxt("Test_Trajectory_UT_0.csv", delimiter=",")
		Fermat_Trajectory(1, "Test", UT_0, None, None)

def main():
	load_UT_Trajectory(1, 3)
	load_UT_Trajectory(2, 3)
	load_UT_Trajectory(1, 1)
	load_UT_Trajectory(2, 1)
	

if __name__ == '__main__':
	main()