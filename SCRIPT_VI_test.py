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

# Crear agente
agent = ValueIterationAgent(env=env, env_vis=env_vis, gamma=1, theta=1e-5, n_test_episodes_per_iteration=200, T_max=200)

# Ejecutar iteración de valor
agent.run_value_iteration(debug=False, performance_check=False, visualize_learning=True, visualize_every=50)

# # Guardar política y valores de los estados
# agent.plot_policy("policy.png")
# agent.plot_state_values("state_values.png")
# agent.plot_avg_return("avg_return.png")

agent.print_policy()
agent.render_episode()