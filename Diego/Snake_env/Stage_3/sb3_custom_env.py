import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from Diego.Snake_env.Stage_3.snakeenv_stage3 import SnakeEnv
import pickle
import zipfile

ALGORITHM = "PPO"
models_dir_stage2 = "Diego/Snake_env/Stage_2/models2/PPO"
models_dir_stage3 = "Diego/Snake_env/Stage_3/models2/PPO"
log_dir = "Diego/Snake_env/Stage_3/logs2"

os.makedirs(models_dir_stage3, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = SnakeEnv()

# Cargar el modelo entrenado en Stage 2
model_path = os.path.join(models_dir_stage2, "1000000.zip")
model = PPO.load(model_path, env=env, device="cpu", tensorboard_log=log_dir)  # usa CPU para MLP (más estable/rápido)

# Re-entrenar en Stage 3
TIMESTEPS = 200_000
NUM_ITERATIONS = 10

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORITHM)
    model.save(os.path.join(models_dir_stage3, f"{TIMESTEPS * i}"))

# Evaluación
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
env.close()