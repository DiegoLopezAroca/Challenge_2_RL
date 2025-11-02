# Import necessary libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

# Set the goal length for the snake
SNAKE_LEN_GOAL = 30

USE_OBSTACLES_STAGE1 = True
NUM_OBSTACLES_STAGE1 = 5
REPORT_EVERY_STEPS = 20_000

# Define the size of the game table for observation and gameplay
##################################################
# CAMBIAMOS EL VALOR DE tableSizeObs Y tableSize #
##################################################
tableSizeObs = 400
tableSize = 400
halfTable = int(tableSize / 2)

# Initialize max score
max_score = 0

###############################################
# Comprobamos si la manzana esta en el cuerpo #
###############################################
def check_apple_on_snake(apple_position, snake_position):
    while apple_position in snake_position:
        apple_position = [random.randrange(1, tableSize // 10) * 10, random.randrange(1, tableSize // 10) * 10]
    return apple_position

# Function to handle collision with apple and update score
def collision_with_apple(apple_position, score):
    # Generate new random apple position within the game table
    apple_position = [random.randrange(1, tableSize // 10) * 10, random.randrange(1, tableSize // 10) * 10]
    score += 1  # Increment the score
    return apple_position, score


# Function to check if the snake head collides with game boundaries
def collision_with_boundaries(snake_head):
    if snake_head[0] >= tableSize or snake_head[0] < 0 or snake_head[1] >= tableSize or snake_head[1] < 0:
        return True
    else:
        return False


# Function to check if the snake collides with itself
def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    else:
        return False

# Define custom snake game environment class inheriting from gym's Env class
class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()

        self.direction_onehot = True

        # Define the action space: 4 discrete actions (left, right, up, down)
        self.action_space = spaces.Discrete(4)

        extra_dim = 4 if self.direction_onehot else 0

        # Dimension del sensor
        extra_dim += 3  # sensores de peligro cercano (frontal + laterales)
        
        # Define the observation space: a Box with size based on snake length goal and other game parameters
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5 + SNAKE_LEN_GOAL +  extra_dim,),
            dtype=np.float32
        )
        
        # Initialize game display and reward variables
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.prev_reward = 0
        self.total_reward = 0
        self.score = 0
        self.max_score = 0

        # contadores para reportes
        self.global_step = 0
        self.episode_idx = 0
        self.ep_len = 0
        self.best_episode_score = 0
        self.best_episode_len = 0

        # Prev actions history (memoria explícita en la observación)
        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

    # Step function that updates the environment after an action is taken
    def step(self, action):
        # Store previous actions
        self.prev_actions.append(action)
        
        # Update the game UI
        # self._update_ui()

        # Calculate previous distance to the apple
        # prev_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        # Metricas previas (para shaping normalizado)
        apple_px = self._nearest_apple()
        prev_euc  = np.linalg.norm(np.array(self.snake_head) - np.array(apple_px))
        prev_bfs  = self._bfs_shortest(self.snake_head, apple_px)
        prev_area = self._flood_area(self.snake_head)
        prev_safe = self._min_body_dist(self.snake_head)

        # Perform the action
        self._take_action(action)

        # Initialize reward
        reward = 0

        # Check if snake eats the apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.apple_position = check_apple_on_snake(self.apple_position, self.snake_position)
            self.snake_position.insert(0, list(self.snake_head))
            reward += 10  # Reward for eating an apple
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # Check for collisions (with boundaries or self)
        terminated = collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position)
        truncated = False

        self.global_step += 1
        self.ep_len += 1

        if terminated:
            reward -= 80.0
            if self.score > self.best_episode_score:
                self.best_episode_score = self.score
                self.best_episode_len = self.ep_len
                print(f"[NEW BEST] ep={self.episode_idx}  score={self.best_episode_score}  len={self.best_episode_len}")

            if REPORT_EVERY_STEPS and (self.global_step % REPORT_EVERY_STEPS == 0):
                print(f"[REPORT] steps={self.global_step}  episodes={self.episode_idx+1}  "
                    f"best_score={self.best_episode_score}  best_len={self.best_episode_len}")
        else:
            # Metricas actuales
            apple_px = self._nearest_apple()
            curr_euc  = np.linalg.norm(np.array(self.snake_head) - np.array(apple_px))
            curr_bfs  = self._bfs_shortest(self.snake_head, apple_px)
            curr_area = self._flood_area(self.snake_head)
            curr_safe = self._min_body_dist(self.snake_head)

            # Límites ajustados a 400x400 px (40x40 celdas)
            BFS_MAX  = 80    # distancia BFS razonable (celdas)
            AREA_MAX = 2000   # área total accesible (40x40)
            SAFE_MAX = 200    # distancia útil al cuerpo (px)

            # Normalización a [-1,1] para comparaciones relativas
            norm_prev_bfs  = -1.0 if prev_bfs is None else self._sym(1.0 - self._norm01(min(prev_bfs,  BFS_MAX), 0, BFS_MAX))
            norm_curr_bfs  = -1.0 if curr_bfs is None else self._sym(1.0 - self._norm01(min(curr_bfs,  BFS_MAX), 0, BFS_MAX))
            norm_prev_area = self._sym(self._norm01(min(prev_area, AREA_MAX), 0, AREA_MAX))
            norm_curr_area = self._sym(self._norm01(min(curr_area, AREA_MAX), 0, AREA_MAX))
            norm_prev_safe = self._sym(self._norm01(min(prev_safe, SAFE_MAX), 0, SAFE_MAX))
            norm_curr_safe = self._sym(self._norm01(min(curr_safe, SAFE_MAX), 0, SAFE_MAX))

            moved_away = (curr_euc > prev_euc + 1e-6)

            # ¿Alejarse estuvo justificado en términos relativos?
            justified = (
                (curr_bfs is not None and prev_bfs is not None and (norm_curr_bfs - norm_prev_bfs) >= +0.01)
                or ((norm_curr_area - norm_prev_area) >= +0.02)
                or ((norm_curr_safe - norm_prev_safe) >= +0.02)
            )

            if moved_away:
                reward += 0.05 if justified else -0.05
            else:
                reward += 0.05
                # Evita encerrarte aunque te acerques
                if curr_area + 10 < prev_area:
                    reward -= 0.05

            # step penalty suave
            reward -= 0.08

        # Information dictionary
        info = {}

        # Create observation of the current state
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # Reset the environment to the initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset game board
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        
        # Reset snake position
        self.snake_position = [
            [halfTable, halfTable],
            [halfTable - 10, halfTable],
            [halfTable - 20, halfTable]
        ]
        
        # Generate random apple position
     
        self.apple_position = [
            random.randrange(1, tableSize // 10) * 10,
            random.randrange(1, tableSize // 10) * 10
        ]

        self.apple_position = check_apple_on_snake(self.apple_position, self.snake_position)
        
        # Obstáculos (vacio en caso de querer añadirlos)
        self.obstacles = set()

        # Grid en celdas (500/10 = 50)
        self.table_w = tableSize // 10
        self.table_h = tableSize // 10

        # Update max score if needed
        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")
        
        # Reset score and snake head position
        self.score = 0
        self.snake_head = [halfTable, halfTable]

        # Initialize direction of the snake
        self.direction = 1

        # Reset reward values
        self.prev_reward = 0
        self.total_reward = 0

        # Create observation of the current state
        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = self._get_obs()

        return observation, {}

    # Render the game visually using OpenCV
    def render(self, mode='human'):
        cv2.imshow('Snake Game', self.img)
        cv2.waitKey(1)
        time.sleep(0.001)  # Add delay between frames to slow down execution

    # Close the OpenCV windows
    def close(self):
        cv2.destroyAllWindows()

    # Update the UI with the current positions of the snake and the apple
    def _update_ui(self):
        # Clear the game board
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        
        # Draw apple on the board
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            -1
        )
        
        # Draw snake on the board
        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                -1
            )
        self.render()

    # Handle actions and update the snake's position
    def _take_action(self, action):
        # Avoid direct opposite movements
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        # Update direction based on the action
        self.direction = action

        # Move the snake head according to the action
        if action == 0:  # Move left
            self.snake_head[0] -= 10
        elif action == 1:  # Move right
            self.snake_head[0] += 10
        elif action == 2:  # Move down
            self.snake_head[1] += 10
        elif action == 3:  # Move up
            self.snake_head[1] -= 10

    def _get_obs(self):
        hx = (self.snake_head[0] / (tableSize/2)) - 1.0
        hy = (self.snake_head[1] / (tableSize/2)) - 1.0
        dx = (self.apple_position[0] - self.snake_head[0]) / (tableSize/2)
        dy = (self.apple_position[1] - self.snake_head[1]) / (tableSize/2)

        length = np.tanh(len(self.snake_position)/20.0)
        pa = np.array([(a/3.0) if a>=0 else -1.0 for a in self.prev_actions], dtype=np.float32)

        base = np.array([hx, hy, dx, dy, length], dtype=np.float32)
        obs = np.concatenate([base, pa])

        if self.direction_onehot:
            oh = np.zeros(4, dtype=np.float32); oh[self.direction] = 1.0
            obs = np.concatenate([obs, oh])

        sensors = self._sensors()
        obs = np.concatenate([obs, sensors])

        return obs

    ############################################
    # NUEVOS HELPERS PARA SHAPING / NAVEGACIÓN #
    ############################################

    def _norm01(self, x, lo, hi):
        """Normalizador genérico: convierte un valor x del rango [lo, hi] a [0, 1].
        Motivo: llevar métricas heterogéneas (BFS, área, distancia de seguridad) a una escala comparable
        y estable para el shaping, independiente del tamaño del tablero."""
        if x is None:
            return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    def _sym(self, x01):
        """Escalado simétrico: convierte [0, 1] en [-1, 1].
        Motivo: mantener consistencia con observation_space [-1, 1] y facilitar umbrales relativos."""
        return 2.0 * x01 - 1.0

    def _grid_pos(self, p):
        """Convierte coordenadas en píxeles a coordenadas de celda (tamaño celda=10px).
        Motivo: los algoritmos de BFS y flood-fill trabajan en la rejilla lógica, no en píxeles."""
        return (p[0] // 10, p[1] // 10)

    def _blocked_set(self):
        """Conjunto de celdas bloqueadas por el cuerpo y los obstáculos.
        Motivo: evitar que BFS/flood atraviesen cuerpo u obstáculos al estimar alcanzabilidad."""
        occ = set((x // 10, y // 10) for x, y in self.snake_position)
        obs = set((x // 10, y // 10) for (x, y) in self.obstacles)
        return occ | obs

    def _nearest_apple(self):
        """Devuelve la posición (px) de la manzana más cercana a la cabeza.
        Motivo: el shaping usa la manzana “objetivo” más relevante (la más cercana). Compatible con Stage 2 (1 apple)
        y Stage 3 (varias): si existe self.apples (lista), la usa; si no, cae a self.apple_position."""
        head = self.snake_head
        if hasattr(self, "apples") and self.apples:
            return min(self.apples, key=lambda ap: np.linalg.norm(np.array(head) - np.array(ap[1])))[1]
        else:
            return self.apple_position

    def _bfs_shortest(self, start_px, goal_px, max_len=2000):
        """Distancia de camino real (en celdas) por BFS desde start hasta goal, respetando celdas bloqueadas.
        Motivo: medir “alcanzabilidad real” a la manzana, más informativa que la distancia euclídea cuando hay cuerpo/obstáculos."""
        W = self.table_w if hasattr(self, 'table_w') else (tableSize // 10)
        H = self.table_h if hasattr(self, 'table_h') else (tableSize // 10)
        start = self._grid_pos(start_px); goal = self._grid_pos(goal_px)
        blocked = self._blocked_set()
        if goal in blocked:  # permitir entrar en la celda de la manzana
            blocked = set(b for b in blocked if b != goal)
        q = deque([(start, 0)])
        seen = {start}
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        while q:
            (x,y), d = q.popleft()
            if (x,y) == goal:
                return d
            if d >= max_len:
                break
            for dx,dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < W and 0 <= ny < H and (nx,ny) not in blocked and (nx,ny) not in seen:
                    seen.add((nx,ny)); q.append(((nx,ny), d+1))
        return None  # no alcanzable

    def _flood_area(self, start_px, limit=2500):
        """Calcula el área accesible (nº de celdas alcanzables) desde start mediante flood-fill.
        Motivo: premiar decisiones que abren espacio de juego y evitar ‘callejones sin salida’."""
        W = self.table_w; H = self.table_h
        start = self._grid_pos(start_px)
        blocked = self._blocked_set()
        if start in blocked:
            blocked = set(b for b in blocked if b != start)
        q = deque([start]); seen = {start}
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        while q and len(seen) < limit:
            x, y = q.popleft()
            for dx,dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < W and 0 <= ny < H and (nx,ny) not in blocked and (nx,ny) not in seen:
                    seen.add((nx,ny)); q.append((nx,ny))
        return len(seen)

    def _min_body_dist(self, head_px):
        """Distancia Manhattan mínima (en píxeles) desde la cabeza al resto del cuerpo.
        Motivo: incentivar márgenes de seguridad y reconocer rodeos seguros que evitan auto-colisiones."""
        head = np.array(head_px)
        if len(self.snake_position) <= 1:
            return 999
        body = np.array(self.snake_position[1:], dtype=np.int32)
        dists = np.abs(body - head).sum(axis=1)  # Manhattan
        return int(dists.min()) if dists.size else 999
    
    #####################################################
    # SENSORES DE PELIGRO CERCANO (FRONTAL + LATERALES) #
    #####################################################
    def _dir_vec(self):
        dir_idx = int(self.direction)  # <-- cast
        return {0: (-10, 0), 1: (10, 0), 2: (0, 10), 3: (0, -10)}[dir_idx]


    def _danger_at(self, cell_px):
        """True si la celda (en píxeles) es peligrosa (muro, obstáculo o cuerpo)."""
        # choque con límites
        if collision_with_boundaries(cell_px):
            return True
        # obstáculos
        if tuple(cell_px) in self.obstacles:
            return True
        # cuerpo (conservador: incluye toda la cola)
        if cell_px in self.snake_position[1:]:
            return True
        return False

    def _sensors(self):
        """
        Devuelve np.array([front, left, right]) con valores en {-1.0, +1.0}:
        +1.0 = hay peligro en esa dirección inmediata
        -1.0 = libre
        """
        dx, dy = self._dir_vec()

        # frente
        f = [self.snake_head[0] + dx, self.snake_head[1] + dy]
        # izquierda (rotar -90º: (-dy, dx))
        lx, ly = -dy, dx
        l = [self.snake_head[0] + lx, self.snake_head[1] + ly]
        # derecha (rotar +90º: (dy, -dx))
        rx, ry = dy, -dx
        r = [self.snake_head[0] + rx, self.snake_head[1] + ry]

        to_sym = lambda b: 1.0 if b else -1.0
        return np.array([to_sym(self._danger_at(f)),
                        to_sym(self._danger_at(l)),
                        to_sym(self._danger_at(r))], dtype=np.float32)
