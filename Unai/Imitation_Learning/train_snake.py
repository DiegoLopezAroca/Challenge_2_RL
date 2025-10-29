import os
import numpy as np
import random
from typing import List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- Config ----------
# Usa el dataset que grabaste con tu SnakeEnv: [(obs_rgb, action_int), ...] por episodio
NPY_PATH = "./Unai/Imitation_Learning/snake_expert_data.npy"   # o "./Unai/snake_expert_data_filtered.npy"
STACK = 4                      # nº de frames apilados (temporal context)
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.1
RANDOM_SEED = 42
IMG_SIZE = 84                  # redimensionado a 84x84
GRAYSCALE = True               # escala de grises para reducir parámetros
NORM_MEAN = 0.5
NORM_STD  = 0.5                # normaliza aprox a [-1,1]

NUM_ACTIONS = 4                # Snake: 0=left,1=right,2=down,3=up

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Utils ----------
def to_grayscale(img: np.ndarray) -> np.ndarray:
    # img: HxWx3 uint8 -> HxWx1 float32 [0,1]
    img_f = img.astype(np.float32) / 255.0
    # RGB -> Gray
    gray = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    return gray[..., None]  # HxWx1

def resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    # Bilinear con torch (rápido y correcto)
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)  # 1xCxHxW
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = t.squeeze(0).numpy().transpose(1,2,0)
    return out

def normalize(img: np.ndarray) -> np.ndarray:
    return (img - NORM_MEAN) / NORM_STD

# ---------- Dataset ----------
class SnakeBCDataset(Dataset):
    """
    Carga el .npy (lista de episodios; cada episodio es lista de (obs_rgb, action_int)).
    Construye ejemplos con stack temporal de K frames dentro de cada episodio.
    Devuelve:
      x: tensor (C,H,W) donde C = (1 ó 3) * STACK
      y: action int en [0..3]
    """
    def __init__(self, npy_path: str, stack: int = 4, grayscale: bool = True,
                 img_size: int = 84, split: str = "train", val_ratio: float = 0.1):
        assert split in {"train", "val"}
        data = np.load(npy_path, allow_pickle=True)  # list[episode] -> list[(obs, action)]
        self.episodes: List[List[Tuple[np.ndarray, int]]] = []
        for ep in data:
            # Asegurar tipos: obs como np.uint8 (H,W,3), action como int
            cleaned = []
            for obs, act in ep:
                if not isinstance(act, (int, np.integer)):
                    # por si vino como np.array([action])
                    act = int(np.array(act).item())
                cleaned.append((obs, act))
            if len(cleaned) > 0:
                self.episodes.append(cleaned)

        # Indexamos por (ep_idx, t_idx)
        all_indices: List[Tuple[int,int]] = []
        for ei, ep in enumerate(self.episodes):
            for ti in range(len(ep)):
                all_indices.append((ei, ti))

        random.shuffle(all_indices)
        val_count = int(len(all_indices) * val_ratio)
        if split == "val":
            self.indices = all_indices[:val_count]
        else:
            self.indices = all_indices[val_count:]

        self.stack = stack
        self.grayscale = grayscale
        self.img_size = img_size
        self.in_channels = (1 if grayscale else 3) * stack

    def __len__(self):
        return len(self.indices)

    def _preproc_one(self, img: np.ndarray) -> np.ndarray:
        # img: HxWx3 RGB uint8
        if self.grayscale:
            img = to_grayscale(img)       # HxWx1 in [0,1]
        else:
            img = img.astype(np.float32)/255.0  # HxWx3
        img = resize_bilinear(img, self.img_size, self.img_size)  # HxWxC
        img = normalize(img)               # (x-mean)/std
        return img.astype(np.float32)

    def __getitem__(self, idx):
        ep_idx, t_idx = self.indices[idx]
        ep = self.episodes[ep_idx]

        # Construir stack de frames [t-K+1 .. t] dentro del episodio (con padding de borde)
        frames = []
        for k in range(self.stack):
            ti = max(0, t_idx - (self.stack - 1 - k))
            frame_k = ep[ti][0]  # obs_rgb
            frames.append(self._preproc_one(frame_k))  # HxWxC

        # Concatenar por canal: (H,W,C*K) -> (C*K,H,W)
        x = np.concatenate(frames, axis=2).transpose(2,0,1)  # (C*K,H,W)
        y = ep[t_idx][1]  # action int

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# ---------- Modelo CNN (clasificación 4 acciones) ----------
class ClassifierCNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256, num_actions: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        # calcular tamaño conv out
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
            n_flat = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)  # logits de 4 acciones
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        logits = self.fc(z)
        return logits  # usar CrossEntropyLoss directamente sobre logits

def make_loaders():
    train_ds = SnakeBCDataset(NPY_PATH, stack=STACK, grayscale=GRAYSCALE,
                              img_size=IMG_SIZE, split="train", val_ratio=VAL_SPLIT)
    val_ds   = SnakeBCDataset(NPY_PATH, stack=STACK, grayscale=GRAYSCALE,
                              img_size=IMG_SIZE, split="val", val_ratio=VAL_SPLIT)
    in_channels = train_ds.in_channels
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, in_channels

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train():
    train_loader, val_loader, in_channels = make_loaders()
    model = ClassifierCNN(in_channels, num_actions=NUM_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        # ---- Train ----
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n_tr = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            tr_loss += loss.item() * bs
            tr_acc += accuracy_from_logits(logits.detach(), y) * bs
            n_tr += bs

        tr_loss /= max(1, n_tr)
        tr_acc  /= max(1, n_tr)

        # ---- Val ----
        model.eval()
        va_loss = 0.0
        va_acc  = 0.0
        n_va = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                bs = x.size(0)
                va_loss += loss.item() * bs
                va_acc  += accuracy_from_logits(logits, y) * bs
                n_va += bs

        va_loss /= max(1, n_va)
        va_acc  /= max(1, n_va)

        print(f"[{epoch}/{EPOCHS}] train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}")

        # Guardar mejor por val_loss
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "in_channels": in_channels,
                "img_size": IMG_SIZE,
                "grayscale": GRAYSCALE,
                "stack": STACK,
                "num_actions": NUM_ACTIONS
            }, "bc_snake.pt")
            print("  ✓ Guardado: bc_snake.pt  (mejor val)")

if __name__ == "__main__":
    train()
