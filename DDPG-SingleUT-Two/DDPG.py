import gym
import gym_foo
import numpy as np
import math
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make('foo-v0')

n_actions = env.action_space.shape[-1]

model = DDPG("MlpPolicy", env, action_noise=None, learning_rate=1e-4, buffer_size=pow(2, 21), batch_size=pow(2, 12), verbose=1, tensorboard_log="./ddpg_SingleUT_Two_tensorboard/")
model.learn(total_timesteps=410000, log_interval=10)
model.save("ddpg_SingleUT_Two")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_SingleUT_Two")

obs = env.reset()
R = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    R += rewards
    if dones==True:
        break
        
    env.render()
print(R)