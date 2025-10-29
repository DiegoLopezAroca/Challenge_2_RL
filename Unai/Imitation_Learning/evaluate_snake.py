import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from snakeenv import SnakeEnv  # usa tu mismo entorno

# ======= Debe coincidir con el entrenamiento =======
GRAYSCALE = True
IMG_SIZE  = 84
STACK     = 4
NORM_MEAN = 0.5
NORM_STD  = 0.5
NUM_ACTIONS = 4  # 0=left,1=right,2=down,3=up

CKPT_PATH = "bc_snake.pt"
EPISODES  = 3
# ====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Modelo (misma arquitectura que training_snake.py) ----------
class ClassifierCNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256, num_actions: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
            n_flat = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        logits = self.fc(z)
        return logits

# ---------- Prepro ----------
def to_grayscale(img: np.ndarray) -> np.ndarray:
    # img RGB uint8 -> HxWx1 float32 [0,1]
    img_f = img.astype(np.float32) / 255.0
    gray = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    return gray[..., None]

def resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)  # 1xCxHxW
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = t.squeeze(0).numpy().transpose(1,2,0)
    return out

def normalize(img: np.ndarray) -> np.ndarray:
    return (img - NORM_MEAN) / NORM_STD

def preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    # rgb: HxWx3 uint8
    if GRAYSCALE:
        img = to_grayscale(rgb)  # HxWx1 in [0,1]
    else:
        img = rgb.astype(np.float32) / 255.0  # HxWx3
    img = resize_bilinear(img, IMG_SIZE, IMG_SIZE)  # HxWxC
    img = normalize(img).astype(np.float32)
    x = img.transpose(2, 0, 1)  # CxHxW
    return x

def get_rgb_from_env(env: SnakeEnv) -> np.ndarray:
    """
    Captura el frame actual del env.
    Tu env dibuja en env.img en BGR, así que convertimos a RGB.
    """
    if getattr(env, "img", None) is None or env.img.size == 0:
        return None
    bgr = env.img
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def main():
    # Cargar checkpoint
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No encuentro el checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)

    # Usa metadatos si están presentes
    in_channels = ckpt.get("in_channels", (1 if GRAYSCALE else 3) * STACK)
    mdl_stack   = ckpt.get("stack", STACK)
    mdl_gray    = ckpt.get("grayscale", GRAYSCALE)
    mdl_imgsz   = ckpt.get("img_size", IMG_SIZE)
    # Asegurar consistencia (si difieren, avisa)
    if (mdl_stack != STACK) or (mdl_gray != GRAYSCALE) or (mdl_imgsz != IMG_SIZE):
        print(f"⚠️ Aviso: los metadatos del checkpoint difieren del script.")
        print(f"  ckpt: stack={mdl_stack}, grayscale={mdl_gray}, img_size={mdl_imgsz}")
        print(f"  script: stack={STACK}, grayscale={GRAYSCALE}, img_size={IMG_SIZE}")

    model = ClassifierCNN(in_channels=in_channels, num_actions=NUM_ACTIONS).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Crear entorno con render humano (OpenCV)
    env = SnakeEnv()

    for ep in range(EPISODES):
        obs, _ = env.reset()
        terminated = False
        truncated = False

        # Inicializa buffer de frames
        frames = deque(maxlen=STACK)

        # Tomar un frame inicial (si todavía no hay dibujo, avanzaremos 1 paso "default")
        rgb = get_rgb_from_env(env)
        if rgb is None:
            # Avanzamos 1 paso siguiendo la dirección inicial del env (normalmente 1=right)
            _ = env.step(getattr(env, "direction", 1))
            rgb = get_rgb_from_env(env)

        x0 = preprocess_rgb(rgb)
        for _ in range(STACK):
            frames.append(x0)

        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            # Montar input (C*K,H,W)
            x_stacked = np.concatenate(list(frames), axis=0)
            x_t = torch.from_numpy(x_stacked).unsqueeze(0).float().to(device)

            with torch.no_grad():
                logits = model(x_t)
                action = int(torch.argmax(logits, dim=1).item())  # 0..3

            # Paso del entorno
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Capturar nuevo frame para el siguiente paso
            rgb = get_rgb_from_env(env)
            if rgb is None:
                break
            x_next = preprocess_rgb(rgb)
            frames.append(x_next)

        print(f"Episodio {ep+1} → Recompensa total: {total_reward:.2f} en {steps} pasos")

    env.close()

if __name__ == "__main__":
    main()
