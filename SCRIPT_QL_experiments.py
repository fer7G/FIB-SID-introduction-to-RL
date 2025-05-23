import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from agents.QLearningAgent import QLearningAgent
from agents.ValueIterationAgent import ValueIterationAgent

def run_experiment_alpha():
    env = gym.make("CliffWalking-v0", is_slippery=True)
    agent_vi = ValueIterationAgent(env, env_vis=None, gamma=0.9, theta=1e-5, n_test_episodes_per_iteration=200, T_max=200)
    agent_vi.run_value_iteration()
    v_star = agent_vi.V.copy()

    alphas = [0.05, 0.2]
    n_episodes = 5000
    errors_by_alpha = {}

    for alpha in alphas:
        print(f"Entrenando Q-learning con alpha = {alpha}")
        env = gym.make("CliffWalking-v0", is_slippery=True)
        agent_q = QLearningAgent(env, env_vis=None, alpha=alpha, gamma=0.9, epsilon=0.1, T_max=200, n_eval_episodes=100)
        agent_q.v_star = v_star
        agent_q.learn(n_episodes=n_episodes, performance_check=False, debug=False)
        errors_by_alpha[alpha] = agent_q.value_error_per_episode

    plt.figure(figsize=(10, 6))
    for alpha, errors in errors_by_alpha.items():
        plt.plot(errors, label=f"α = {alpha}")
    plt.title("Error promedio normalizado de $V(s)$ respecto a $v_*(s)$\nsegún distintos valores de α")
    plt.xlabel("Episodio")
    plt.ylabel("Error promedio normalizado")
    plt.legend(title="Tasa de aprendizaje")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/value_error_multiple_alphas.png", dpi=300)
    plt.show()

def run_experiment_decay():
    env = gym.make("CliffWalking-v0", is_slippery=True)
    agent_vi = ValueIterationAgent(env, env_vis=None, gamma=0.95, theta=1e-5, n_test_episodes_per_iteration=200, T_max=200)
    agent_vi.run_value_iteration()
    v_star = agent_vi.V.copy()

    configs = {
        "Sin decay": (False, False),
        "Decay ε": (True, False),
        "Decay α": (False, True),
        "Decay ambos": (True, True),
    }

    n_episodes = 2000
    error_curves = {}

    for name, (decay_eps, decay_alpha) in configs.items():
        print(f"\nEntrenando: {name}")
        env = gym.make("CliffWalking-v0", is_slippery=True)
        alpha = 0.2 if decay_alpha else 0.05
        epsilon = 0.2 if decay_eps else 0.1
        agent = QLearningAgent(env, env_vis=None, alpha=alpha, gamma=0.95, epsilon=epsilon, T_max=200, n_eval_episodes=100)
        agent.v_star = v_star
        agent.learn(n_episodes=n_episodes, performance_check=False, debug=False,
                    decay_epsilon=decay_eps, decay_alpha=decay_alpha)
        error_curves[name] = agent.value_error_per_episode

    plt.figure(figsize=(10, 6))
    for name, errors in error_curves.items():
        plt.plot(errors, label=name)
    plt.title("Error normalizado de $V(s)$ respecto a $v_*(s)$ bajo distintas estrategias de decay")
    plt.xlabel("Episodio")
    plt.ylabel("Error promedio normalizado")
    plt.grid(True)
    plt.legend(title="Configuración")
    plt.tight_layout()
    plt.savefig("images/q_learning_decay_comparison.png", dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['alpha', 'decay'], required=True, help="Nombre del experimento")
    args = parser.parse_args()

    if args.exp == 'alpha':
        run_experiment_alpha()
    elif args.exp == 'decay':
        run_experiment_decay()

if __name__ == '__main__':
    main()
