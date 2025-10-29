import os
import cv2
import numpy as np
from PIL import Image
from snakeenv import SnakeEnv  # <-- usa tu mismo entorno

# ===================== Config =====================
num_episodios = 1
save_npy = "./Unai/Imitation_Learning/snake_expert_data.npy"
save_frames = True
frames_dir = "./Unai/Imitation_Learning/snake_frames"

# Mapeo de acciones del env: 0=left, 1=right, 2=down, 3=up
ACTION_NAMES = ["left", "right", "down", "up"]

# Códigos de teclas en OpenCV (flechas) y alternativas WASD
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 81, 82, 83, 84
KEY_A, KEY_W, KEY_D, KEY_S = ord('a'), ord('w'), ord('d'), ord('s')

def key_to_action(key, last_action):
    # Flechas
    if key == KEY_LEFT:  return 0
    if key == KEY_RIGHT: return 1
    if key == KEY_DOWN:  return 2
    if key == KEY_UP:    return 3
    # WASD (por si las flechas no llegan en tu SO)
    if key == KEY_A: return 0
    if key == KEY_D: return 1
    if key == KEY_S: return 2
    if key == KEY_W: return 3
    # Sin tecla: mantener dirección anterior
    return last_action

def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ===================== Main =====================
if save_frames:
    os.makedirs(frames_dir, exist_ok=True)

env = SnakeEnv()  # tu mismo entorno, sin tocar
expert_data = []  # lista de episodios; cada episodio: [(obs_rgb, action_int), ...]

for ep in range(num_episodios):
    obs, _ = env.reset()
    done = False
    truncated = False
    terminated = False

    # Dirección inicial del env es 1 (derecha) en tu SnakeEnv
    last_action = 1

    frames_episode = []
    ep_dir = os.path.join(frames_dir, f"ep{ep}")
    if save_frames:
        os.makedirs(ep_dir, exist_ok=True)
        txt_path = os.path.join(ep_dir, f"acciones_ep{ep}.txt")
        with open(txt_path, "w") as f:
            f.write("step action_name action_id\n")

    print(f"Episodio {ep+1}/{num_episodios}")

    # Forzar primer frame para poder grabar desde el paso 0
    # (tu env dibuja en _update_ui() durante step; para el frame inicial usamos el estado actual)
    # Generamos una imagen inicial a partir del buffer actual del env si existe
    # Si estuviera vacío, tras el primer step ya se sincroniza.
    if getattr(env, "img", None) is None or env.img.size == 0:
        # Hacemos un paso "nulo" de dibujo llamando a los métodos internos
        # pero sin cambiar estado: el env pinta al hacer step -> así que arrancamos sin grabar,
        # y tras el primer step ya grabamos correctamente. Para mantener el patrón de tu CarRacing,
        # grabaremos el frame anterior a cada acción a partir de ahora.
        pass

    # Para el esquema (frame, action) estilo CarRacing:
    # - Leemos tecla -> decidimos action
    # - Grabamos el frame *anterior* con esa action
    # - Llamamos env.step(action) -> env.img se actualiza
    # - Repetimos
    step = 0
    # Obtener un frame inicial (si el env ya tiene algo en img, lo usamos; si no, el primer append será tras el primer step)
    prev_frame_rgb = None
    if getattr(env, "img", None) is not None and env.img.size != 0:
        prev_frame_rgb = bgr_to_rgb(env.img.copy())

    while not (terminated or truncated):
        # Leer teclado desde cualquier ventana OpenCV
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC o 'q' para salir del episodio
            truncated = True
            break

        action = key_to_action(key, last_action)
        # El entorno ya evita giros de 180° internamente en _take_action
        # así que le podemos mandar la acción directamente.
        if prev_frame_rgb is not None:
            frames_episode.append((prev_frame_rgb, int(action)))
            if save_frames:
                with open(txt_path, "a") as f:
                    f.write(f"{step} {ACTION_NAMES[action]} {int(action)}\n")

        # Avanzar el entorno
        obs, reward, terminated, truncated, info = env.step(action)

        # Capturar el frame que el env acaba de dibujar
        if getattr(env, "img", None) is not None and env.img.size != 0:
            prev_frame_rgb = bgr_to_rgb(env.img.copy())
        else:
            prev_frame_rgb = None

        last_action = action
        step += 1

    # Si el episodio terminó y hay un último frame dibujado, opcionalmente podrías agregarlo con la última acción
    # (comentado para mantener el mismo patrón de "obs+action antes del step")
    # if prev_frame_rgb is not None and step > 0:
    #     frames_episode.append((prev_frame_rgb, int(last_action)))

    expert_data.append(frames_episode)

    # Guardar frames en PNG
    if save_frames:
        for s, (frame_obs, act) in enumerate(frames_episode):
            Image.fromarray(frame_obs).save(f"{ep_dir}/step{s:05d}.png")

    # Guardar el npy (tras cada episodio)
    np.save(save_npy, np.array(expert_data, dtype=object), allow_pickle=True)
    print(f"Episodio {ep+1} guardado. Total episodios: {len(expert_data)}")

env.close()
print(f"Guardado {len(expert_data)} episodios en {save_npy}")
if save_frames:
    print(f"Frames y txt por episodio guardados en '{frames_dir}'")
