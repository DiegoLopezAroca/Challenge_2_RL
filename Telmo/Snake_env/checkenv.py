from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from snakeenv import SnakeEnv

env = SnakeEnv()
check_env(env, warn=True)
