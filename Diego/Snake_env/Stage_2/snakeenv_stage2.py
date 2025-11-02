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
        apple_position = [random.randrange(1, tableSize // 10) * 10,
                          random.randrange(1, tableSize // 10) * 10]
    return apple_position

# Function to handle collision with apple and update score
def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, tableSize // 10) * 10,
                      random.randrange(1, tableSize // 10) * 10]
    score += 1
    return apple_position, score

# Function to check if the snake head collides with game boundaries (WALLS)
def collision_with_boundaries(snake_head):
    return (snake_head[0] >= tableSize or snake_head[0] < 0 or
            snake_head[1] >= tableSize or snake_head[1] < 0)

# Function to check if the snake collides with itself
def collision_with_self(snake_position):
    snake_head = snake_position[0]
    return snake_head in snake_position[1:]

# Define custom snake game environment class inheriting from gym's Env class
class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()

        self.direction_onehot = True
        self.action_space = spaces.Discrete(4)

        extra_dim = (4 if self.direction_onehot else 0) + 3  # onehot + 3 sensores
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5 + SNAKE_LEN_GOAL + extra_dim,), dtype=np.float32
        )

        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.prev_reward = 0.0
        self.total_reward = 0.0
        self.score = 0
        self.max_score = 0

        # contadores
        self.global_step = 0
        self.episode_idx = 0
        self.ep_len = 0
        self.best_episode_score = 0
        self.best_episode_len = 0

        # memoria de acciones
        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        # timeout y “retirada justificada”
        self.steps_since_food = 0
        self.max_steps_no_food = 300
        self.justified_streak = 0

    # Step function
    def step(self, action):
        self.prev_actions.append(action)

        # métricas previas
        apple_px = self._nearest_apple()
        prev_euc  = np.linalg.norm(np.array(self.snake_head) - np.array(apple_px))
        prev_bfs  = self._bfs_shortest(self.snake_head, apple_px)
        prev_area = self._flood_area(self.snake_head)
        prev_safe = self._min_body_dist(self.snake_head)

        # acción
        self._take_action(action)

        reward = 0.0

        # comida
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.apple_position = check_apple_on_snake(self.apple_position, self.snake_position)
            self.snake_position.insert(0, list(self.snake_head))
            reward += 40.0         # ↑ recompensa por comer
            self.steps_since_food = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # terminales: MURO o CUERPO
        terminated = collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position)
        truncated = False

        self.global_step += 1

        if self.global_step % 10000 == 0:
            print(
                f"[{self.global_step}] "
                f"best_ep_score={self.best_episode_score} "
                f"max_score={self.max_score} "
                f"ep={self.episode_idx} len={self.ep_len}",
                flush=True
            )
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
            # métricas actuales
            apple_px = self._nearest_apple()
            curr_euc  = np.linalg.norm(np.array(self.snake_head) - np.array(apple_px))
            curr_bfs  = self._bfs_shortest(self.snake_head, apple_px)
            curr_area = self._flood_area(self.snake_head)
            curr_safe = self._min_body_dist(self.snake_head)

            # Límites para 40x40 celdas
            BFS_MAX  = 80     # camino razonable
            AREA_MAX = 1600   # 40*40
            SAFE_MAX = 200    # px

            # normalización
            norm_prev_bfs  = -1.0 if prev_bfs is None else self._sym(1.0 - self._norm01(min(prev_bfs,  BFS_MAX), 0, BFS_MAX))
            norm_curr_bfs  = -1.0 if curr_bfs is None else self._sym(1.0 - self._norm01(min(curr_bfs,  BFS_MAX), 0, BFS_MAX))
            norm_prev_area = self._sym(self._norm01(min(prev_area, AREA_MAX), 0, AREA_MAX))
            norm_curr_area = self._sym(self._norm01(min(curr_area, AREA_MAX), 0, AREA_MAX))
            norm_prev_safe = self._sym(self._norm01(min(prev_safe, SAFE_MAX), 0, SAFE_MAX))
            norm_curr_safe = self._sym(self._norm01(min(curr_safe, SAFE_MAX), 0, SAFE_MAX))

            # mejoras relativas
            dbfs  = 0.0 if (curr_bfs is None or prev_bfs is None) else (norm_curr_bfs - norm_prev_bfs)
            darea = (norm_curr_area - norm_prev_area)
            dsafe = (norm_curr_safe - norm_prev_safe)

            moved_away = (curr_euc > prev_euc + 1e-6)

            # Umbrales de justificación (más estrictos con paredes)
            JUST_BFS   = +0.02
            JUST_AREA  = +0.03
            JUST_SAFE  = +0.03

            justified = (dbfs >= JUST_BFS) or (darea >= JUST_AREA) or (dsafe >= JUST_SAFE)

            # suavizado
            if justified:
                self.justified_streak = min(self.justified_streak + 1, 3)
            else:
                self.justified_streak = max(self.justified_streak - 1, 0)

            # potencial: acercarse suma, alejarse resta poco
            delta = (prev_euc - curr_euc) / 10.0
            reward += float(np.clip(delta, -0.1, 0.1))

            # penaliza alejarse si NO está justificado y sin racha
            if moved_away and (not justified) and (self.justified_streak == 0):
                reward -= 0.03

            # no te encierres reduciendo espacio drásticamente
            if (curr_area + 10) < prev_area:
                reward -= 0.02

            # coste por paso bajo
            reward -= 0.02

            # timeout por “vagueo”
            self.steps_since_food += 1
            if self.steps_since_food >= self.max_steps_no_food:
                truncated = True
                reward -= 5.0

        info = {}
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # Reset the environment to the initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.snake_position = [
            [halfTable, halfTable],
            [halfTable - 10, halfTable],
            [halfTable - 20, halfTable]
        ]
        self.apple_position = [
            random.randrange(1, tableSize // 10) * 10,
            random.randrange(1, tableSize // 10) * 10
        ]
        self.apple_position = check_apple_on_snake(self.apple_position, self.snake_position)

        self.obstacles = set()

        self.table_w = tableSize // 10
        self.table_h = tableSize // 10

        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")
            
        self.score = 0
        self.snake_head = [halfTable, halfTable]
        self.direction = 1

        self.prev_reward = 0.0
        self.total_reward = 0.0

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        self.steps_since_food = 0
        self.justified_streak = 0

        observation = self._get_obs()
        return observation, {}

    # Render
    def render(self, mode='human'):
        cv2.imshow('Snake Game', self.img)
        cv2.waitKey(1)
        time.sleep(0.001)

    def close(self):
        cv2.destroyAllWindows()

    def _update_ui(self):
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        cv2.rectangle(self.img,
                      (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10),
                      (0, 0, 255), -1)
        for position in self.snake_position:
            cv2.rectangle(self.img,
                          (position[0], position[1]),
                          (position[0] + 10, position[1] + 10),
                          (0, 255, 0), -1)
        self.render()

    # Actions
    def _take_action(self, action):
        # evitar giros 180º
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        self.direction = action

        if action == 0:      # left
            self.snake_head[0] -= 10
        elif action == 1:    # right
            self.snake_head[0] += 10
        elif action == 2:    # down
            self.snake_head[1] += 10
        elif action == 3:    # up
            self.snake_head[1] -= 10

    # Observation
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
    # HELPERS PARA SHAPING / NAVEGACIÓN        #
    ############################################
    def _norm01(self, x, lo, hi):
        if x is None:
            return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    def _sym(self, x01):
        return 2.0 * x01 - 1.0

    def _grid_pos(self, p):
        return (p[0] // 10, p[1] // 10)

    def _blocked_set(self):
        occ = set((x // 10, y // 10) for x, y in self.snake_position)
        obs = set((x // 10, y // 10) for (x, y) in self.obstacles)
        return occ | obs

    def _nearest_apple(self):
        head = self.snake_head
        if hasattr(self, "apples") and self.apples:
            return min(self.apples, key=lambda ap: np.linalg.norm(np.array(head) - np.array(ap[1])))[1]
        else:
            return self.apple_position

    def _bfs_shortest(self, start_px, goal_px, max_len=3000):
        W = self.table_w if hasattr(self, 'table_w') else (tableSize // 10)
        H = self.table_h if hasattr(self, 'table_h') else (tableSize // 10)
        start = self._grid_pos(start_px); goal = self._grid_pos(goal_px)
        blocked = self._blocked_set()
        if goal in blocked:
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
        return None

    def _flood_area(self, start_px, limit=5000):
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
        dir_idx = int(self.direction)
        return {0: (-10, 0), 1: (10, 0), 2: (0, 10), 3: (0, -10)}[dir_idx]

    def _danger_at(self, cell_px):
        """True si la celda (px) es peligrosa (muro, obstáculo o cuerpo)."""
        if collision_with_boundaries(cell_px):  # muros
            return True
        if tuple(cell_px) in self.obstacles:    # obstáculos
            return True
        if cell_px in self.snake_position[1:]:  # cuerpo
            return True
        return False

    def _sensors(self):
        """
        np.array([front, left, right]) en {-1.0, +1.0}:
        +1.0 = peligro inmediato, -1.0 = libre
        """
        dx, dy = self._dir_vec()

        f = [self.snake_head[0] + dx, self.snake_head[1] + dy]     # frente
        lx, ly = -dy, dx                                           # izquierda
        l = [self.snake_head[0] + lx, self.snake_head[1] + ly]
        rx, ry = dy, -dx                                           # derecha
        r = [self.snake_head[0] + rx, self.snake_head[1] + ry]

        to_sym = lambda b: 1.0 if b else -1.0
        return np.array([to_sym(self._danger_at(f)),
                         to_sym(self._danger_at(l)),
                         to_sym(self._danger_at(r))], dtype=np.float32)
