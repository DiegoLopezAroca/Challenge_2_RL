import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from collections import deque

# Grid moves in steps of 10 px on a square board
CELL = 10
BOARD = 500
HALF = BOARD // 2
SNAKE_LEN_GOAL = 30

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, max_steps=5000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # 4 discrete actions: 0=left, 1=right, 2=down, 3=up
        self.action_space = spaces.Discrete(4)

        # Observation: [head_x, head_y, apple_dx, apple_dy, snake_length] + prev_actions (len=SNAKE_LEN_GOAL)
        obs_len = 5 + SNAKE_LEN_GOAL
        # Finite bounds to satisfy SB3; values always within these ranges
        low = np.full((obs_len,), -float(BOARD), dtype=np.float32)
        high = np.full((obs_len,), float(BOARD), dtype=np.float32)
        # previous actions will be in [-1, 3]; still within low/high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = None  # will be set in reset(seed=)
        self.window_name = "Snake"

        # Internal buffers initialized in reset()
        self.img = None
        self.snake_position = None
        self.snake_head = None
        self.apple_position = None
        self.direction = None
        self.prev_actions = None
        self.score = 0
        self.max_score = 0
        self.steps = 0

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng, _ = gym.utils.seeding.np_random(seed)

        self.img = np.zeros((BOARD, BOARD, 3), dtype=np.uint8)

        # Snake of length 3 centered horizontally
        self.snake_position = [
            [HALF, HALF],
            [HALF - CELL, HALF],
            [HALF - 2 * CELL, HALF],
        ]
        self.snake_head = list(self.snake_position[0])
        self.direction = 1  # initial direction: right

        self.apple_position = self._spawn_apple()
        self.prev_actions = deque([-1] * SNAKE_LEN_GOAL, maxlen=SNAKE_LEN_GOAL)

        self.steps = 0
        self.score = 0

        obs = self._get_obs().astype(np.float32)
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        self.prev_actions.append(int(action))
        self.steps += 1

        # Prevent immediate 180° turns
        action = self._sanitize_action(action)

        # Previous distance (for shaping)
        prev_distance = np.linalg.norm(np.array(self.snake_head, dtype=np.int32) - np.array(self.apple_position, dtype=np.int32))

        # Move head
        if action == 0:   # left
            self.snake_head[0] -= CELL
        elif action == 1: # right
            self.snake_head[0] += CELL
        elif action == 2: # down
            self.snake_head[1] += CELL
        elif action == 3: # up
            self.snake_head[1] -= CELL

        # Insert new head
        self.snake_position.insert(0, list(self.snake_head))

        reward = 0.0
        terminated = False
        truncated = False

        # Eat apple?
        if self.snake_head == self.apple_position:
            reward += 10.0
            self.score += 1
            self.apple_position = self._spawn_apple()
        else:
            # normal move: remove tail
            self.snake_position.pop()

        # Collisions
        if self._collision_with_boundaries(self.snake_head) or self._collision_with_self(self.snake_position):
            reward -= 10.0
            terminated = True

        # Distance shaping (if not terminated)
        if not terminated:
            curr_distance = np.linalg.norm(np.array(self.snake_head, dtype=np.int32) - np.array(self.apple_position, dtype=np.int32))
            reward += 1.0 if curr_distance < prev_distance else -1.0

        # Small step penalty
        reward -= 0.1

        # Time-limit truncation to keep episodes bounded
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs().astype(np.float32)
        info = {"score": self.score}

        if self.render_mode is not None:
            self._render_frame()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        # Gymnasium calls this for some render modes; keep for compatibility
        if self.render_mode == "rgb_array":
            return self._draw_frame(copy=True)
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    # ---------- Helpers ----------
    def _get_obs(self):
        head_x, head_y = self.snake_head
        apple_dx = self.apple_position[0] - head_x
        apple_dy = self.apple_position[1] - head_y
        snake_length = len(self.snake_position)
        base = [head_x, head_y, apple_dx, apple_dy, float(snake_length)]
        return np.array(base + list(self.prev_actions), dtype=np.float32)

    def _collision_with_boundaries(self, head):
        return head[0] < 0 or head[0] >= BOARD or head[1] < 0 or head[1] >= BOARD

    def _collision_with_self(self, snake_position):
        head = snake_position[0]
        return head in snake_position[1:]

    def _spawn_apple(self):
        # Ensure apple is on grid and not on the snake
        while True:
            ax = int(self._rng.integers(1, BOARD // CELL)) * CELL
            ay = int(self._rng.integers(1, BOARD // CELL)) * CELL
            if [ax, ay] not in self.snake_position:
                return [ax, ay]

    def _sanitize_action(self, action):
        # avoid 180 turns
        # 0<->1 (left<->right), 2<->3 (down<->up)
        if (action == 0 and self.direction == 1) or \
           (action == 1 and self.direction == 0) or \
           (action == 2 and self.direction == 3) or \
           (action == 3 and self.direction == 2):
            action = self.direction  # keep going current direction
        self.direction = action
        return action

    def _draw_frame(self, copy=False):
        if self.img is None:
            self.img = np.zeros((BOARD, BOARD, 3), dtype=np.uint8)
        self.img.fill(0)

        # Apple
        cv2.rectangle(self.img,
                      (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + CELL, self.apple_position[1] + CELL),
                      (0, 0, 255), -1)

        # Snake
        for x, y in self.snake_position:
            cv2.rectangle(self.img, (x, y), (x + CELL, y + CELL), (0, 255, 0), -1)

        return self.img.copy() if copy else self.img

    def _render_frame(self):
        frame = self._draw_frame()
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(int(1000 / self.metadata["render_fps"]))
