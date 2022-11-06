import numpy as np
import globe
import matplotlib.pyplot as plt
import gym
import gym_foo

def plot(frame_idx, rewards):
    plt.figure()
    plt.title('Step %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig('Exhaustive_Result/rewards.png', format='png')
    plt.close()

##########################################################################
maxStep = 41

globe._init()
env = gym.make('foo-v0')
env.seed(100)

total_record = []
def main():
	rewards = []
	received_energy_per_step = []
	harvested_energy_per_step = []

	for steps in range(maxStep):
		print("The current steps: "+str(steps))
		max_reward = 0
		harvest_energy = 0
		energy_per_step = 0
		action = []
		for i in range(int(1e6)):
			action = np.random.rand(20)
			next_state, reward, done, received_energy = env.step(action, steps)
			if reward/received_energy > max_reward:
				max_reward = reward/received_energy
				energy_per_step = received_energy
				harvest_energy = reward

		rewards.append(max_reward)
		received_energy_per_step.append(energy_per_step)
		harvested_energy_per_step.append(harvest_energy)
		if steps == maxStep - 1:
			np.savetxt("Exhaustive_Result/rewards.csv", rewards, delimiter=',')
			np.savetxt("Exhaustive_Result/received_energy_per_step.csv", received_energy_per_step, delimiter=',')
			np.savetxt("Exhaustive_Result/harvested_energy_per_step.csv", harvested_energy_per_step, delimiter=',')

	total_record.append(np.sum(rewards))

	print("rewards："+str(total_record[0]))
	print("harvested rewards："+str(np.sum(harvested_energy_per_step)/np.sum(received_energy_per_step)))

	np.savetxt("Exhaustive_Result/total_rewards.csv", total_record, delimiter=',')

if __name__ == '__main__':
	main()