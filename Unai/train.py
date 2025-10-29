import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from snakeenv import SnakeEnv

ALGO = "PPO"
models_dir = f"models/{ALGO}"
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Create env (no rendering during training)
env = SnakeEnv(render_mode=None, max_steps=5000)

# PPO with MLP policy (works well on vector obs). Use policy_kwargs if you want a bigger net.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2
)

TIMESTEPS = 10_000
NUM_ITERS = 10

for i in range(1, NUM_ITERS + 1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGO)
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# Quick eval (no render)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()
