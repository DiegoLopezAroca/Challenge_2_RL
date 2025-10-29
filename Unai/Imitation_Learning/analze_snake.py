import numpy as np
import os

# === CONFIGURACIÓN ===
# Ruta a tus datos originales del grabador de Snake
data_path = "./Unai/Imitation_Learning/snake_expert_data.npy"

# Ruta de salida para los episodios filtrados
save_path = "./Unai/Imitation_Learning/snake_expert_data_filtered.npy"

# Índices de los episodios "buenos" que quieres conservar
good_eps = [0]   # puedes añadir más, p. ej. [0, 2, 4]

# ======================
if not os.path.exists(data_path):
    raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

# Cargar datos (lista de episodios -> cada episodio: lista de (obs_rgb, action_int))
data = np.load(data_path, allow_pickle=True)

print(f"Se cargaron {len(data)} episodios del dataset original.")

# Filtrar episodios buenos
filtered = [data[i] for i in good_eps if i < len(data)]

# Guardar resultado
np.save(save_path, np.array(filtered, dtype=object), allow_pickle=True)

print(f"Guardado '{save_path}' con {len(filtered)} episodios filtrados.")
