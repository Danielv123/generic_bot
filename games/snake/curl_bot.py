# Contrastive unsupervised reinfocement learning to play snake

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import keyboard

config = {
    "board_size": 10,
    "action_space": 4,
    "state_space": 1000,
}

directions = {
    0: (0, -1),  # Up
    1: (1, 0),   # Right
    2: (0, 1),   # Down
    3: (-1, 0),  # Left
}

def main():
    # Create a simple snake game
    snake = [(3, 2), (3, 1), (3, 0)]
    food = (np.random.randint(0, config["board_size"]), np.random.randint(0, config["board_size"]))
    game_over = False
    score = 0

    def get_game_state():
        # Return array of pixels on the board
        board = np.zeros((config["board_size"], config["board_size"], 3))
        # Add the snake
        # Add snake body
        for segment in snake[1:]:
            board[segment[0], segment[1]] = [1, 0, 0]
        # Add snake head with different color
        board[snake[0][0], snake[0][1]] = [0, 0, 1]
        # Add the food
        board[food[0], food[1]] = [0, 1, 0]
        return board
    
    def make_decision(state):
        return np.random.randint(0, config["action_space"])

    def take_action(action):
        # Move the snake
        new_head = (snake[0][0] + directions[action][0], snake[0][1] + directions[action][1])
        snake.insert(0, new_head)
        # Check if the snake has eaten the food
        nonlocal food
        if new_head == food:
            food = (np.random.randint(0, config["board_size"]), np.random.randint(0, config["board_size"]))
            score += 1
        else:
            snake.pop()

    def check_game_over():
        # Check if the snake has hit the edge of the board or itself
        if snake[0][0] < 0 or snake[0][0] >= config["board_size"] or snake[0][1] < 0 or snake[0][1] >= config["board_size"]:
            return True
        # Check if the snake has hit itself
        if snake[0] in snake[1:]:
            return True
        return False
    
    while not game_over and keyboard.is_pressed("esc") == False:
        # Get the current state of the game
        state = get_game_state()

        # Make a decision based on the state
        action = make_decision(state)

        # Take the action in the game
        take_action(action)

        # Check if the game is over
        game_over = check_game_over()

        # Render the game
        if not game_over:
            render(get_game_state(), score)
    else:
        cv2.destroyAllWindows()

def render(board, score):
    # Render the board
    cv2.imshow("Snake", board)
    cv2.waitKey(1000)

if __name__ == "__main__":
    main()