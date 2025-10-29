import sys
import time
from stable_baselines3 import PPO
from snakeenv import SnakeEnv

# Usage: python play.py models/PPO/100000.zip
model_path = sys.argv[1] if len(sys.argv) > 1 else "models/PPO/100000.zip"

# Human render mode
env = SnakeEnv(render_mode="human", max_steps=5000)

model = PPO.load(model_path, env=env)

obs, info = env.reset(seed=42)
done = False
truncated = False

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        time.sleep(0.5)
        obs, info = env.reset()
