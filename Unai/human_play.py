import cv2
import numpy as np
import random
import time

# --- Game settings ---
CELL = 10
BOARD = 500
HALF = BOARD // 2
SPEED = 100  # smaller = faster (milliseconds per move)

# --- Initialize ---
snake = [[HALF, HALF], [HALF - CELL, HALF], [HALF - 2 * CELL, HALF]]
direction = 1  # 0=left, 1=right, 2=down, 3=up
apple = [random.randrange(1, BOARD // CELL) * CELL, random.randrange(1, BOARD // CELL) * CELL]
score = 0

# --- Key mapping ---
KEYS = {
    81: 0,  # left arrow
    82: 3,  # up arrow
    83: 1,  # right arrow
    84: 2   # down arrow
}

# --- Helper functions ---
def new_apple():
    while True:
        a = [random.randrange(1, BOARD // CELL) * CELL, random.randrange(1, BOARD // CELL) * CELL]
        if a not in snake:
            return a

def collision(head):
    x, y = head
    if x < 0 or y < 0 or x >= BOARD or y >= BOARD:
        return True
    if head in snake[1:]:
        return True
    return False

# --- Main loop ---
cv2.namedWindow("Snake - Arrow keys to move | ESC to quit")

while True:
    # --- Draw frame ---
    img = np.zeros((BOARD, BOARD, 3), dtype=np.uint8)
    cv2.rectangle(img, (apple[0], apple[1]), (apple[0] + CELL, apple[1] + CELL), (0, 0, 255), -1)
    for x, y in snake:
        cv2.rectangle(img, (x, y), (x + CELL, y + CELL), (0, 255, 0), -1)
    cv2.putText(img, f"Score: {score}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Snake - Arrow keys to move | ESC to quit", img)

    # --- Handle input ---
    key = cv2.waitKey(SPEED) & 0xFF
    if key == 27:  # ESC quits
        break
    elif key in KEYS:
        new_dir = KEYS[key]
        # Prevent 180° turn
        if not ((direction == 0 and new_dir == 1) or (direction == 1 and new_dir == 0) or
                (direction == 2 and new_dir == 3) or (direction == 3 and new_dir == 2)):
            direction = new_dir

    # --- Move snake ---
    head = list(snake[0])
    if direction == 0: head[0] -= CELL
    elif direction == 1: head[0] += CELL
    elif direction == 2: head[1] += CELL
    elif direction == 3: head[1] -= CELL

    snake.insert(0, head)

    # --- Check apple collision ---
    if head == apple:
        score += 1
        apple = new_apple()
    else:
        snake.pop()

    # --- Check collisions ---
    if collision(head):
        cv2.putText(img, "GAME OVER", (BOARD//3, BOARD//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.imshow("Snake - Arrow keys to move | ESC to quit", img)
        cv2.waitKey(2000)
        break

cv2.destroyAllWindows()
