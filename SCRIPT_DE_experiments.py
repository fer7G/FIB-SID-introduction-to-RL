import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from agents import ValueIterationAgent, DirectEstimationAgent, QLearningAgent

def experiment_return_vs_experience():
    sample_sizes = [200, 500, 1000, 2000, 5000, 10000, 20000]
    returns_per_sample_size = {}
    max_iter = 0

    for episodes in sample_sizes:
        print(f"\n--- Estimando p con {episodes} episodios ---")
        env = gym.make("CliffWalking-v0", is_slippery=True)
        agent = DirectEstimationAgent(env, env_vis=None, gamma=1, theta=1e-3, n_test_episodes_per_iteration=200, T_max=200)
        agent.gather_experience(num_episodes=episodes)
        agent.run_value_iteration(performance_check=True)
        returns_per_sample_size[episodes] = agent.avg_return_per_iteration
        max_iter = max(max_iter, len(agent.avg_return_per_iteration))

    plt.figure(figsize=(10, 6))
    for episodes, returns in returns_per_sample_size.items():
        padded_returns = returns + [np.nan] * (max_iter - len(returns))
        plt.plot(padded_returns, label=f"{episodes} episodios")

    plt.title("Retorno promedio por iteración de valor\nsegún número de episodios de experiencia aleatoria")
    plt.xlabel("Iteración de valor")
    plt.ylabel("Retorno promedio")
    plt.legend(title="Experiencia usada")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experimento_retornos.png", dpi=300)
    plt.show()


def experiment_policy_vs_optimal():
    optimal_policy_str = [
        '↑','→','→','→','→','→','→','→','→','→','→','→',
        '↑','→','→','→','→','→','→','→','→','→','→','→',
        '↑','↑','↑','↑','↑','↑','↑','↑','↑','↑','↑','→',
        '←','X','X','X','X','X','X','X','X','X','X','G'
    ]
    action_symbols = ['↑', '→', '↓', '←']
    symbol_to_index = {sym: i for i, sym in enumerate(action_symbols)}
    optimal_policy = [symbol_to_index.get(sym, -1) for sym in optimal_policy_str]
    sample_sizes = [10000, 20000]

    def plot_policy_comparison(policy, title, filename):
        bg_colors = np.ones((4, 12))
        texts = np.full((4, 12), '', dtype=object)
        for s in range(48):
            row, col = divmod(s, 12)
            if 37 <= s <= 46:
                texts[row, col] = 'X'
                bg_colors[row, col] = 0.5
            elif s == 47:
                texts[row, col] = 'G'
                bg_colors[row, col] = 0.5
            else:
                a = policy[s]
                texts[row, col] = action_symbols[a]
                bg_colors[row, col] = 0.7 if a == optimal_policy[s] else 0.2

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(bg_colors, cmap='RdYlGn', vmin=0, vmax=1)
        for i in range(4):
            for j in range(12):
                ax.text(j, i, texts[i, j], ha='center', va='center', fontsize=14, color='black')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    for episodes in sample_sizes:
        print(f"Ejecutando con {episodes} episodios de experiencia aleatoria...")
        env = gym.make("CliffWalking-v0", is_slippery=True)
        agent = DirectEstimationAgent(env, env_vis=None, gamma=1, theta=1e-3, n_test_episodes_per_iteration=200, T_max=200)
        agent.gather_experience(num_episodes=episodes)
        agent.run_value_iteration()
        plot_policy_comparison(agent.policy, f"Política aprendida con {episodes} episodios", f"policy_comparison_{episodes}.png")


def experiment_value_error_vs_v_star():
    env = gym.make("CliffWalking-v0", is_slippery=True)
    agent_opt = ValueIterationAgent(env, env_vis=None, gamma=1, theta=1e-5, n_test_episodes_per_iteration=200, T_max=200)
    agent_opt.run_value_iteration()
    v_star = agent_opt.V.copy()
    min_v, max_v = np.min(v_star), np.max(v_star)
    range_v = max_v - min_v

    sample_sizes = list(range(500, 20001, 500))
    errors = []

    for episodes in sample_sizes:
        print(f"Estimando con {episodes} episodios de experiencia aleatoria...")
        env = gym.make("CliffWalking-v0", is_slippery=True)
        agent = DirectEstimationAgent(env, env_vis=None, gamma=1, theta=1e-3, n_test_episodes_per_iteration=200, T_max=200)
        agent.gather_experience(num_episodes=episodes)
        agent.run_value_iteration()

        error = 0
        count = 0
        for s in range(agent.n_states):
            if 37 <= s <= 46 or s == 47:
                continue
            error += abs(agent.V[s] - v_star[s])
            count += 1

        avg_error_normalized = (error / count) / range_v
        errors.append(avg_error_normalized)

    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, errors)
    plt.title("Error promedio normalizado en $V(s)$ respecto a $v_*(s)$")
    plt.xlabel("Nº de episodios aleatorios usados para estimar $\hat{p}$")
    plt.ylabel("Error promedio normalizado")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("value_error_normalized.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de estimación directa")
    parser.add_argument("--exp", type=str, choices=["returns", "policy", "vstar"], required=True,
                        help="Nombre del experimento a ejecutar: returns, policy o vstar")
    args = parser.parse_args()

    if args.exp == "returns":
        experiment_return_vs_experience()
    elif args.exp == "policy":
        experiment_policy_vs_optimal()
    elif args.exp == "vstar":
        experiment_value_error_vs_v_star()
