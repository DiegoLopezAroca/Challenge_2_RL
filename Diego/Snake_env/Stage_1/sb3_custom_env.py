import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from Diego.Snake_env.Stage_1.snakeenv import SnakeEnv
import pickle
import zipfile

ALGORITHM = "PPO"
models_dir = f"Diego/Snake_env/Stage_1/models2/{ALGORITHM}"
log_dir = "Diego/Snake_env/Stage_1/logs2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
env = SnakeEnv()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 50000
NUM_ITERATIONS = 5  # Adjust according to your needs

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=ALGORITHM
    )
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()

