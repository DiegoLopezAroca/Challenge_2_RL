import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from Diego.Snake_env.Stage_2.snakeenv_stage2 import SnakeEnv
import pickle
import zipfile

ALGORITHM = "PPO"
models_dir_stage1 = "Diego/Snake_env/Stage_1/models/PPO"
models_dir_stage2 = "Diego/Snake_env/Stage_2/models/PPO"
log_dir = "Diego/Snake_env/Stage_2/logs"

os.makedirs(models_dir_stage2, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = SnakeEnv()

# Cargar el modelo entrenado en Stage 1
model_path = os.path.join(models_dir_stage1, "500000.zip")
model = PPO.load(model_path, env=env, device="cpu", tensorboard_log=log_dir)  # usa CPU para MLP (más estable/rápido)

# Re-entrenar en Stage 2
TIMESTEPS = 100_000
NUM_ITERATIONS = 5

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORITHM)
    model.save(os.path.join(models_dir_stage2, f"{TIMESTEPS * i}"))

# Evaluación
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
env.close()