import numpy as np
import os

# === CONFIGURACIÓN ===
# Archivo original con los datos grabados desde Snake
data_path = "./Unai/Imitation_Learning/snake_expert_data.npy"

# Archivo limpio (filtrado)
save_path = "./Unai/Imitation_Learning/snake_expert_data_filtered.npy"

# Índices de los episodios “buenos” que queremos conservar
good_eps = [0]  # Ejemplo: [0, 2, 4]

# ======================

if not os.path.exists(data_path):
    raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

# Cargar datos originales (lista de episodios, cada uno: [(frame_rgb, action_int), ...])
data = np.load(data_path, allow_pickle=True)
print(f"Dataset original cargado con {len(data)} episodios.")

# Filtrar los episodios buenos
filtered = [data[i] for i in good_eps if i < len(data)]

# Guardar los episodios limpios
np.save(save_path, np.array(filtered, dtype=object), allow_pickle=True)
print(f"Guardado '{save_path}' con {len(filtered)} episodios limpios.")
