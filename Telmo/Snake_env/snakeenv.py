# snakeenv.py
# -------------------------------------------------------
# Entorno personalizado del juego Snake compatible con Gymnasium
# con soporte para curriculum learning (tamaño y velocidad configurables)
# -------------------------------------------------------

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

# Longitud objetivo del historial de acciones previas
SNAKE_LEN_GOAL = 30
# Rango máximo global del espacio de observación (constante)
GLOBAL_OBS_LIMIT = 500


# ==========================================================
# Funciones auxiliares
# ==========================================================

def collision_with_apple(apple_position, score, table_size):
    apple_position = [
        random.randrange(1, table_size // 10) * 10,
        random.randrange(1, table_size // 10) * 10
    ]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head, table_size):
    if snake_head[0] >= table_size or snake_head[0] < 0 or snake_head[1] >= table_size or snake_head[1] < 0:
        return True
    else:
        return False


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    else:
        return False


# ==========================================================
# Clase del entorno personalizado
# ==========================================================

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, table_size=500, render_speed=0.01):
        super(SnakeEnv, self).__init__()

        # Parámetros de dificultad
        self.tableSize = table_size
        self.render_speed = render_speed
        self.halfTable = int(self.tableSize / 2)

        # Espacios de acción y observación (fijos para evitar errores)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-GLOBAL_OBS_LIMIT,
            high=GLOBAL_OBS_LIMIT,
            shape=(5 + SNAKE_LEN_GOAL,),
            dtype=np.float64
        )

        # Inicialización
        self.img = np.zeros((self.tableSize, self.tableSize, 3), dtype='uint8')
        self.prev_reward = 0
        self.total_reward = 0
        self.score = 0
        self.max_score = 0

    def step(self, action):
        self.prev_actions.append(action)
        self._update_ui()

        prev_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        self._take_action(action)
        reward = 0

        # Comer manzana
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score, self.tableSize)
            self.snake_position.insert(0, list(self.snake_head))
            reward += 10
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # Colisiones
        terminated = collision_with_boundaries(self.snake_head, self.tableSize) or collision_with_self(self.snake_position)
        truncated = False

        if terminated:
            reward -= 10
        else:
            current_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
            if current_distance < prev_distance:
                reward += 1
            else:
                reward -= 1

        reward -= 0.1  # Pequeña penalización por paso

        # Observación
        head_x, head_y = self.snake_head
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float64)

        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.img = np.zeros((self.tableSize, self.tableSize, 3), dtype='uint8')

        self.snake_position = [
            [self.halfTable, self.halfTable],
            [self.halfTable - 10, self.halfTable],
            [self.halfTable - 20, self.halfTable]
        ]

        self.apple_position = [
            random.randrange(1, self.tableSize // 10) * 10,
            random.randrange(1, self.tableSize // 10) * 10
        ]

        if self.score > self.max_score:
            self.max_score = self.score
            print(f"Nuevo máximo score: {self.max_score}")

        self.score = 0
        self.snake_head = [self.halfTable, self.halfTable]
        self.direction = 1
        self.prev_reward = 0
        self.total_reward = 0

        head_x, head_y = self.snake_head
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float64)
        return observation, {}

    def render(self, mode='human'):
        cv2.imshow('Snake Game', self.img)
        cv2.waitKey(1)
        time.sleep(self.render_speed)

    def close(self):
        cv2.destroyAllWindows()

    def _update_ui(self):
        self.img = np.zeros((self.tableSize, self.tableSize, 3), dtype='uint8')
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            -1
        )
        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                -1
            )
        self.render()

    def _take_action(self, action):
        # Evita movimientos opuestos directos
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        self.direction = action
        if action == 0:  # izquierda
            self.snake_head[0] -= 10
        elif action == 1:  # derecha
            self.snake_head[0] += 10
        elif action == 2:  # abajo
            self.snake_head[1] += 10
        elif action == 3:  # arriba
            self.snake_head[1] -= 10
