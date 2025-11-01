# sb3_custom_env.py
# -------------------------------------------------------
# Entrenamiento del agente PPO con curriculum learning progresivo
# -------------------------------------------------------

import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from snakeenv import SnakeEnv

ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Etapas del curriculum: cada una más difícil
curriculum = [
    {"table_size": 200, "render_speed": 0.05},  # Nivel fácil
    {"table_size": 300, "render_speed": 0.03},  # Nivel medio
    {"table_size": 400, "render_speed": 0.02},  # Nivel difícil
    {"table_size": 500, "render_speed": 0.01},  # Nivel experto
]

TIMESTEPS = 10000
model = None
results = []

for i, level in enumerate(curriculum, start=1):
    print(f"\n=== Entrenando nivel {i}/{len(curriculum)} ===")
    print(f"Tamaño del tablero: {level['table_size']}, velocidad: {level['render_speed']}")

    env = SnakeEnv(table_size=level["table_size"], render_speed=level["render_speed"])

    if model is None:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    else:
        model.set_env(env)

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{ALGORITHM}_level{i}")
    model.save(f"{models_dir}/{ALGORITHM}_level{i}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    results.append((i, level["table_size"], mean_reward, std_reward))
    print(f"Nivel {i} -> Recompensa media: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()

print("\n✅ Entrenamiento completo con curriculum learning finalizado.\n")

# Mostrar resumen de resultados
print("Resumen de desempeño por nivel:")
for i, size, mean, std in results:
    print(f" Nivel {i} | Tamaño {size} | Recompensa media: {mean:.2f} ± {std:.2f}")
