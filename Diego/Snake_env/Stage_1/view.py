import os
import time
import cv2
from stable_baselines3 import PPO
from Diego.Snake_env.Stage_1.snakeenv import SnakeEnv  # <-- ajusta si tu clase está en otro archivo

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
ALGORITHM = "PPO"
models_dir_stage3 = "Diego/Snake_env/Stage_1/models2/PPO"
model_path = os.path.join(models_dir_stage3, "500000.zip")  # <-- cambia al nombre del modelo que quieras visualizar

# Crear entorno (modo visual)
env = SnakeEnv()

# Cargar el modelo entrenado
model = PPO.load(model_path, env=env, device="cpu")

# -------------------------------
# VISUALIZACIÓN DEL AGENTE
# -------------------------------
obs, info = env.reset()
done = False
trunc = False
resta = 1
print("Reproduciendo el modelo... Cierra la ventana para salir.")
while resta:
    # Obtener acción del modelo (determinista para comportamiento estable)
    action, _ = model.predict(obs, deterministic=True)

    # Aplicar la acción
    obs, reward, done, trunc, info = env.step(action)

    # Renderizar (si tu entorno ya dibuja en step(), no hace falta llamar a env.render())
    # env.render()

    # Controlar velocidad (opcional)
    time.sleep(0.03)

    # Reiniciar episodio si termina
    if done or trunc:
        obs, info = env.reset()
        resta -= 1

# Cierra ventanas al salir
env.close()
