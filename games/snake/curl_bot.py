# Contrastive unsupervised reinfocement learning to play snake

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import keyboard
from collections import deque
import random
import matplotlib.pyplot as plt
from collections import defaultdict

config = {
    "board_size": 5,
    "action_space": 4,
    "state_space": 1000,
}

directions = {
    0: (0, -1),  # Up
    1: (1, 0),   # Right
    2: (0, 1),   # Down
    3: (-1, 0),  # Left
}

class QNetwork(nn.Module):
    def __init__(self):
        # Initialize the Q-Network architecture
        super(QNetwork, self).__init__()
        
        # Input layer - flattens the 3 color channels x board size squared
        input_size = 3 * config["board_size"] * config["board_size"]
        
        # Simple feedforward network
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, config["action_space"])
        
    def forward(self, x):
        # Use contiguous() before view() to ensure proper memory layout
        x = x.contiguous().view(-1, 3 * config["board_size"] * config["board_size"])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
class SnakeAgent:
    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        
        # Add tracking metrics
        self.metrics = {
            'scores': [],
            'rewards': [],
            'epsilons': [],
            'moving_avg_score': []
        }
    
    def log_metrics(self, episode_score, episode_rewards):
        self.metrics['scores'].append(episode_score)
        self.metrics['rewards'].append(sum(episode_rewards))
        self.metrics['epsilons'].append(self.epsilon)
        
        # Calculate moving average score (last 100 episodes)
        window_size = min(100, len(self.metrics['scores']))
        moving_avg = sum(self.metrics['scores'][-window_size:]) / window_size
        self.metrics['moving_avg_score'].append(moving_avg)
    
    def plot_metrics(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot scores
        ax1.plot(self.metrics['scores'], label='Score', alpha=0.6)
        ax1.plot(self.metrics['moving_avg_score'], label='Moving Average (100)', color='red')
        ax1.set_title('Scores over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend()
        
        # Plot rewards
        ax2.plot(self.metrics['rewards'], label='Total Reward', color='green')
        ax2.set_title('Rewards over Episodes')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
        
        # Plot epsilon
        ax3.plot(self.metrics['epsilons'], label='Epsilon', color='orange')
        ax3.set_title('Epsilon over Episodes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, config["action_space"]-1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Then convert to tensors
        states = torch.FloatTensor(states).permute(0, 3, 1, 2)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def main():
    agent = SnakeAgent()
    num_episodes = 1000

    for episode in range(num_episodes):
        # Reset game state
        snake = [(3, 2), (3, 1), (3, 0)]
        food = (np.random.randint(0, config["board_size"]), 
                np.random.randint(0, config["board_size"]))
        game_over = False
        score = 0
        steps = 0
        episode_rewards = []  # Track rewards for this episode

        while not game_over and not keyboard.is_pressed("esc"):
            state = get_game_state(snake, food)
            action = agent.get_action(state)
            
            # Calculate distances and take action
            old_distance = manhattan_distance(snake[0], food)  # Changed to Manhattan distance
            snake, food, score, self_collision = take_action(action, snake, food, score)
            new_distance = manhattan_distance(snake[0], food)  # Changed to Manhattan distance
            
            # Calculate reward
            reward = 0
            if check_game_over(snake):
                reward = -10
                game_over = True
            elif snake[0] == food:
                reward = 20  # Increased reward for getting food
            else:
                # Simplified distance-based reward
                if new_distance < old_distance:
                    reward = 1  # Reward for moving closer
                else:
                    reward = -1  # Penalty for moving away
            
            if self_collision:
                reward = -10  # Increased penalty for self collision
            
            episode_rewards.append(reward)  # Track reward
                
            if not game_over:
                next_state = get_game_state(snake, food)
                # Store experience in memory
                agent.memory.append((state, action, reward, next_state, game_over))
                
            # Train the agent
            agent.train()
            
            steps += 1
            if steps % 100 == 0:  # Update target network periodically
                agent.target_network.load_state_dict(agent.q_network.state_dict())

            # if not game_over:
            #     render(get_game_state(snake, food), score)
        
        # Log metrics for this episode
        agent.log_metrics(score, episode_rewards)
        
        # Print progress and plot every 100 episodes
        print(f"Episode {episode}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
        # if (episode + 1) % 100 == 0:
        #     agent.plot_metrics()
        #     plt.pause(0.1)  # Small pause to allow plot to update

    # Final plot
    agent.plot_metrics()
    plt.show()

def get_game_state(snake, food):
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

def take_action(action, snake, food, score):
    # Move the snake
    new_head = (snake[0][0] + directions[action][0], snake[0][1] + directions[action][1])
    # Don't allow the snake to move backwards
    # if new_head[0] == snake[1][0] and new_head[1] == snake[1][1]:
    #     return snake, food, score, True
    snake.insert(0, new_head)
    # Check if the snake has eaten the food
    if new_head == food:
        food = (np.random.randint(0, config["board_size"]), np.random.randint(0, config["board_size"]))
        score += 1
    else:
        snake.pop()
    return snake, food, score, False

def check_game_over(snake):
    # Check if the snake has hit the edge of the board or itself
    if snake[0][0] < 0 or snake[0][0] >= config["board_size"] or snake[0][1] < 0 or snake[0][1] >= config["board_size"]:
        return True
    # Check if the snake has hit itself
    # if snake[0] in snake[1:]:
    #     return True
    return False

def render(board, score):
    # Render the board
    cv2.imshow("Snake", board)
    cv2.waitKey(100)

# Add this new helper function
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

if __name__ == "__main__":
    main()