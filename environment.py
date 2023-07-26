import numpy as np 
import gym
import random
from gym.envs.registration import register
import matplotlib.pyplot as plt
import time
import seaborn as sns

if 'FrozenLake8x8NotSlippery-v0' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['FrozenLake8x8NotSlippery-v0']

register(
    id='FrozenLake8x8NotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=1000,
    reward_threshold=0.8196
)

n_episodes = 100000
gamma = 1.0
env = gym.make("FrozenLake8x8NotSlippery-v0")