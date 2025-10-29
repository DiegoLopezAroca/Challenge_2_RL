import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from snakeenv import SnakeEnv

ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Crear entorno
env = SnakeEnv()

# Instanciar agente
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
NUM_ITERATIONS = 10

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=ALGORITHM)
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# Evaluación
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
