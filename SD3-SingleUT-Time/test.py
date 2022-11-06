import numpy as np
import torch
import gym
import gym_foo
import argparse
import os
import utils
import random

import SD3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_policy(policy, env_name, seed, eval_episodes=1, eval_cnt=None):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	Rewards = []
	Received_Energy_List = []
	Harvested_Energy_ratio_List = []
	for episode_idx in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			print(action)
			next_state, reward, done, Received_Energy = eval_env.step(action)

			Rewards.append(reward)
			Received_Energy_List.append(Received_Energy)
			Harvested_Energy_ratio_List.append(reward/Received_Energy)

			state = next_state


	print("[{}] Evaluation over {} episodes, Rewards: {} Harvest_Energy: {}".format(eval_cnt, eval_episodes, np.sum(Rewards), np.sum(Rewards)/np.sum(Received_Energy_List)))
	
	return Rewards, Received_Energy_List, Harvested_Energy_ratio_List


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", default="./logs")
	parser.add_argument("--policy", default="SD3")
	parser.add_argument("--env", default='foo-v0')
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--start-steps", default=1e5, type=int, help='Number of steps for the warm-up stage using random policy')
	parser.add_argument("--eval-freq", default=8100, type=int, help='Number of steps per evaluation')
	parser.add_argument("--steps", default=1458000, type=int, help='Maximum number of steps')

	parser.add_argument("--discount", default=0.99, help='Discount factor')
	parser.add_argument("--tau", default=0.005, help='Target network update rate')                    
	
	parser.add_argument("--actor-lr", default=1e-5, type=float)     
	parser.add_argument("--critic-lr", default=1e-5, type=float)    
	parser.add_argument("--hidden-sizes", default='400,300', type=str)  
	parser.add_argument("--batch-size", default=pow(2, 10), type=int)      # Batch size for both actor and critic

	parser.add_argument("--save-model", action="store_true", default=True)        # Save model and optimizer parameters
	parser.add_argument("--load-model", default="model")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--expl-noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--policy-noise", default=0.2, type=float)              # Noise added to target policy during critic update
	parser.add_argument("--noise-clip", default=0.5, type=float)                # Range to clip target policy noise

	parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')

	parser.add_argument('--beta', default='best', help='The parameter beta in softmax')
	parser.add_argument('--num-noise-samples', type=int, default=100, help='The number of noises to sample for each next_action')
	parser.add_argument('--imps', type=int, default=0, help='Whether to use importance sampling for gaussian noise when calculating softmax values')
	
	args = parser.parse_args()

	print("---------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
	print("---------------------------------------")
	# outdir = "checkpoints"
	# if args.save_model and not os.path.exists("{}/models".format(outdir)):
	# 	os.makedirs("{}/models".format(outdir))

	Rewards = []
	Harvest_Energy = []

	env = gym.make(args.env)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.cuda.manual_seed_all(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	min_action = float(env.action_space.low[0])#-max_action
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
		"actor_lr": args.actor_lr,
		"critic_lr": args.critic_lr,
		"device": device,
	}

	if args.policy == "SD3":
		env_beta_map = {
			'foo-v0': 5
		}

		kwargs['beta'] = env_beta_map[args.env] if args.beta == 'best' else float(args.beta)
		kwargs['with_importance_sampling'] = args.imps
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs['num_noise_samples'] = args.num_noise_samples

		policy = SD3.SD3(**kwargs)

	if args.load_model != "":
		policy.load("./checkpoints/models/{}".format(args.load_model))


	eval_cnt = 0
	
	eval_return, Received_Energy, eval_Harvested_Energy = eval_policy(policy, args.env, args.seed, eval_episodes=1, eval_cnt=eval_cnt)
	eval_cnt += 1

	state, done = env.reset(), False


	np.savetxt("Evaluated_Harvested_Energy.csv", eval_Harvested_Energy, delimiter=',')
	np.savetxt("Total_HarvestEnergy.txt", [np.sum(eval_return)/np.sum(Received_Energy)])
