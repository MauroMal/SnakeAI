import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer, Linear_QNet
from helper import plot

MAX_MEMORY = 100_000 # Limit memory size for experience replay
BATCH_SIZE = 1000  # Number of samples per training step
LR = 0.001  # Learning rate for the neural network

class Agent:
    # Initializes the agent for training the Snake AI
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # pop_left
        self.model = Linear_QNet(11, 256, 3) # Neural network with input, hidden, and output layers
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # Handles model training


    def get_state(self, game):
        head = game.snake[0] # Snake's head position
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Determine current movement direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x, # Food Left
            game.food.x > game.head.x, # Food Right
            game.food.y < game.head.y, # Food Up
            game.food.y > game.head.y, # Food Down
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store experience for future training
        self.memory.append((state, action, reward, next_state, done)) # pop_left if MAX_MEMORY is reached

    def train_long_memory(self):
        # Train using a batch of past experiences
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory # Use all available experiences if not enough samples


        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # Neural network predicts best action
            move = torch.argmax(prediction).item() # Choose action with highest value
            final_move[move] = 1 # Convert to one-hot encoding

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record: # Save model if new high score
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, 'Record:', record)

            # Update score plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()