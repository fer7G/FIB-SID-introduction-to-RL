import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from agents import ValueIterationAgent, DirectEstimationAgent, QLearningAgent

# Crear entorno de entrenamiento (sin render)
env = gym.make("CliffWalking-v0", is_slippery=True, render_mode="none")
# Crear entorno de visualización
env_vis = gym.make("CliffWalking-v0", is_slippery=True, render_mode="human")
env.metadata['render_fps'] = 120

# Crear agente Q-Learning
agent = QLearningAgent(env=env, env_vis=env_vis, alpha=0.05, gamma=1, epsilon=0.1, T_max=200)

# Aprender política
agent.learn(n_episodes=3000, performance_check=False, debug=False, visualize_learning=True, visualize_every=300)

# Imprimir política resultante
agent.print_policy()

# Guardar política y valores de los estados
# agent.plot_policy("policy.png")
# agent.plot_state_values("state_values.png")
# agent.plot_avg_return("avg_return.png")
# agent.plot_state_action_values("state_action_values.png")
