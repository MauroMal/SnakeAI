# SnakeAI
Snake AI Game With Deep Reinforcement Learning

## Python Game Agent

This project is a simple game simulation written in Python where an agent interacts with a game environment. It was built for learning purposes and as a way to experiment with basic AI decision-making. The goal was to separate the logic into clean modules and make it easy to understand and extend.

The main agent is written in `agent.py`, and it's designed to observe the current state of the game, make a decision using logic defined in `model.py`, and send an action back to the game. The game itself is defined in `game.py`, and any helper functions (like calculations or formatting) are stored in `helper.py`.

## What It's Doing

- When you run `agent.py`, it creates a new game instance and starts a loop.
- Each loop, the agent checks the current state of the game.
- Based on that state, it uses a model to figure out the best move.
- Then it applies the move to the game, and the game updates accordingly.
- This repeats until the game ends.

The way the code is split up makes it easy to plug in a different model or change the rules of the game. You could even use this as a starting point for trying out reinforcement learning or some other ML method.

## Files in This Project

- `agent.py`: Main file that runs the game and controls the agent.
- `game.py`: Handles the game logic, including state updates and win/loss conditions.
- `model.py`: Contains the logic for how the agent chooses actions. This could be simple rules or a more complex AI later on.
- `helper.py`: Utility functions that support the other files (e.g., data processing, formatting).

## How to Run It

Make sure you have Python 3 installed. Then just run this command from the terminal:

python3 agent.py

Everything should start from there. You don't need to pass in any arguments or install anything extra unless you decide to add libraries later.
