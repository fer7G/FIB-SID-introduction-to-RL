import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class QLearningAgent:

    def __init__(self, env, env_vis, alpha=0.05, gamma=1.0, epsilon=0.1, T_max=200, n_eval_episodes=200):
        self.env = env
        self.env_vis = env_vis
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.T_max = T_max
        self.n_eval_episodes = n_eval_episodes
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=int)
        self.avg_return_per_episode = []
        self.v_star = None # Valor óptimo de cada estado, esto se establece desde fuera calculándolo con el ValueIterationAgent
        self.value_error_per_episode = []

    def render_episode(self):
        """
        Ejecuta y renderiza un episodio con la política actual.
        """
        state, _ = self.env_vis.reset()
        self.env_vis.render()
        done = False
        steps = 0
        while not done and steps < self.T_max:
            action = self.policy[state]
            state, reward, done, truncated, _ = self.env_vis.step(action)
            self.env_vis.render()
            steps += 1
            if reward == -100:
                break

    def _track_value_error(self):
        if self.v_star is None:
            return
        v_est = np.max(self.Q, axis=1)
        error = 0
        count = 0
        for s in range(self.n_states):
            if 37 <= s <= 46 or s == 47:
                continue
            error += abs(v_est[s] - self.v_star[s])
            count += 1
        range_v = np.max(self.v_star) - np.min(self.v_star)
        avg_error_normalized = error / count / range_v
        self.value_error_per_episode.append(avg_error_normalized)


    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state])

    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode_return = 0
        steps = 0

        while not done and steps < self.T_max:
            action = self.epsilon_greedy(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            best_next_action = np.argmax(self.Q[next_state])
            td_target = reward + self.gamma * self.Q[next_state, best_next_action]
            self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])

            state = next_state
            episode_return += reward
            steps += 1

            if reward == -100:
                break

        return episode_return

    def run_eval_episode(self):
        state, _ = self.env.reset()
        done = False
        episode_return = 0
        steps = 0
        while not done and steps < self.T_max:
            action = self.policy[state]
            state, reward, done, truncated, _ = self.env.step(action)
            episode_return += reward
            steps += 1
            if reward == -100:
                break
        return episode_return

    def check_policy_performance(self):
        returns = [self.run_eval_episode() for _ in range(self.n_eval_episodes)]
        return np.mean(returns)

    def learn(self, n_episodes=5000, performance_check=False, debug=False, decay_epsilon=False, decay_alpha=False, visualize_learning=False, visualize_every=10):
        for i in range(n_episodes):
            self.run_episode()
            self.extract_policy()
            self._track_value_error()

            if performance_check:
                avg_return = self.check_policy_performance()
                self.avg_return_per_episode.append(avg_return)

            if debug:
                ret = f"{avg_return:.2f}" if performance_check else "N/A"
                print(f"Episodio {i}: retorno promedio = {ret}")
                self.print_state_values()
                self.print_policy()

            # Visualizar el aprendizaje
            if visualize_learning and i % visualize_every == 0:
                print(f"\n--- Visualizando episodio en iteración {i} ---")
                self.render_episode()

            if decay_epsilon:
                self.epsilon = max(0.01, self.epsilon * 0.995)  # límite inferior para no dejar de explorar completamente

            # Decaimiento exponencial de alpha
            if decay_alpha:
                self.alpha = max(0.05, self.alpha * 0.995)


    def extract_policy(self):
        for s in range(self.n_states):
            if (37 <= s <= 46) or (s == 47):
                continue
            self.policy[s] = np.argmax(self.Q[s])

    def print_state_values(self):
        """
        Imprime el valor actual de cada estado (máximo Q) en forma de rejilla 4x12.
        """
        n_rows, n_cols = 4, 12
        print("Valores de los estados:")
        for i in range(n_rows):
            row = ""
            for j in range(n_cols):
                state = i * n_cols + j
                if 37 <= state <= 46:
                    row += "X     "
                elif state == 47:
                    row += "G     "
                else:
                    value = np.max(self.Q[state])
                    row += f"{value:5.1f} "
            print(row)
        print("\n")

    def plot_state_action_values(self, filename="state_action_values.png"):
        """
        Muestra un gráfico 4x12 donde cada celda tiene 4 subcasillas (↑,→,↓,←)
        con colores estilo mapa de calor según los valores Q(s, a).
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Valores Q(s, a) por estado y acción (↑→↓←)")

        q_min = np.min(self.Q)
        q_max = np.max(self.Q)
        norm = plt.Normalize(vmin=q_min, vmax=q_max)
        cmap = plt.cm.coolwarm

        for s in range(self.n_states):
            row, col = divmod(s, 12)

            if 37 <= s <= 46:
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='black'))
                ax.text(col + 0.5, row + 0.5, "X", ha="center", va="center", color="white", fontsize=12)
                continue
            elif s == 47:
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='gray'))
                ax.text(col + 0.5, row + 0.5, "G", ha="center", va="center", color="white", fontsize=12)
                continue

            q_up, q_right, q_down, q_left = self.Q[s]

            # Coordenadas relativas
            x, y = col, row
            # UP (arriba izquierda)
            ax.add_patch(plt.Rectangle((x, y), 0.5, 0.5, color=cmap(norm(q_up))))
            ax.text(x + 0.25, y + 0.25, f"{q_up:.1f}", ha="center", va="center", fontsize=7, color="black")
            # RIGHT (arriba derecha)
            ax.add_patch(plt.Rectangle((x + 0.5, y), 0.5, 0.5, color=cmap(norm(q_right))))
            ax.text(x + 0.75, y + 0.25, f"{q_right:.1f}", ha="center", va="center", fontsize=7, color="black")
            # DOWN (abajo derecha)
            ax.add_patch(plt.Rectangle((x + 0.5, y + 0.5), 0.5, 0.5, color=cmap(norm(q_down))))
            ax.text(x + 0.75, y + 0.75, f"{q_down:.1f}", ha="center", va="center", fontsize=7, color="black")
            # LEFT (abajo izquierda)
            ax.add_patch(plt.Rectangle((x, y + 0.5), 0.5, 0.5, color=cmap(norm(q_left))))
            ax.text(x + 0.25, y + 0.75, f"{q_left:.1f}", ha="center", va="center", fontsize=7, color="black")

            # Borde general de la celda
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='gray'))

        # Leyenda del color
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        # cbar.set_label("Valor Q")

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


    def plot_state_values(self, filename="state_values.png"):
        """
        Guarda una imagen 4x12 de los valores (máximo Q) de los estados en formato PNG.
        """
        grid = np.zeros((4, 12))
        for s in range(self.n_states):
            row, col = divmod(s, 12)
            if 37 <= s <= 46:
                grid[row, col] = -100
            elif s == 47:
                grid[row, col] = 0
            else:
                grid[row, col] = np.max(self.Q[s])

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(grid, cmap="coolwarm", vmin=np.min(grid), vmax=np.max(grid))

        for i in range(4):
            for j in range(12):
                state = i * 12 + j
                if 37 <= state <= 46:
                    text = "X"
                elif state == 47:
                    text = "G"
                else:
                    text = f"{grid[i,j]:.1f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=14)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Valores estimados de los estados (max Q)")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def print_policy(self):
        action_symbols = ['↑', '→', '↓', '←']
        n_rows, n_cols = 4, 12
        print("Política actual:")
        for i in range(n_rows):
            row = ""
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx == 47:
                    row += "G  "
                elif 37 <= idx <= 46:
                    row += "X  "
                else:
                    row += f"{action_symbols[self.policy[idx]]}  "
            print(row)
        print()

    def plot_policy(self, filename="policy.png"):
        symbols = ['↑', '→', '↓', '←']
        grid = np.full((4, 12), '', dtype=object)

        for s in range(self.n_states):
            row, col = divmod(s, 12)
            if 37 <= s <= 46:
                grid[row, col] = 'X'
            elif s == 47:
                grid[row, col] = 'G'
            else:
                grid[row, col] = symbols[self.policy[s]]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(np.ones((4, 12)), cmap="gray", vmin=0, vmax=1)

        for i in range(4):
            for j in range(12):
                ax.text(j, i, grid[i, j], ha='center', va='center', fontsize=14, color='black')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Política aprendida")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_avg_return(self, filename="avg_return.png"):
        plt.figure(figsize=(8, 4))
        plt.plot(self.avg_return_per_episode)
        plt.title("Retorno promedio por episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Retorno promedio")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
