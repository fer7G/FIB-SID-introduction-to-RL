import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

class DirectEstimationAgent:

    def __init__(self, env, env_vis, gamma, theta, n_test_episodes_per_iteration, T_max):
        self.env = env                                                      # Entorno Gynasium (MDP)
        self.env_vis = env_vis                                              # Entorno de visualización
        self.n_states = self.env.observation_space.n                        # Número de estados
        self.n_actions = self.env.action_space.n                            # Número de acciones
        self.gamma = gamma                                                  # Factor de descuento
        self.theta = theta                                                  # Umbral de convergencia
        self.V = np.zeros(self.n_states)                                    # Valores de los estados (inicialmente a 0)
        self.policy = np.zeros(self.n_states, dtype=int)                    # Política greedy del agente
        self.n_test_episodes_per_iteration = n_test_episodes_per_iteration  # Número de episodios que se ejecutan en cada iteración para probar la política conseguida
        self.T_max = T_max
        self.policy_changes_per_iteration = []                              # Número de cambios en la política en cada iteración
        self.avg_return_per_iteration = []                                  # Retorno promedio por iteración
        # Almacena el número de veces que se ha llegado a s' desde s tomando a, y hemos recibido r.
        self.transition_reward_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

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

    def record_transition(self, s, a, r, s_):
        """
        Registra la transición (s, a, r, s') en el modelo.
        """
        self.transition_reward_counts[s][a][(s_, r)] += 1

    def compute_empirical_p(self, s, a):
        """
        Para un par estado-acción, nos da cada probabilidad de obtener cada (s', r).
        """
        total = sum(self.transition_reward_counts[s][a].values())
        if total == 0:
            return []
        transitions = []
        for (s_, r), count in self.transition_reward_counts[s][a].items():
            prob = count / total
            transitions.append((prob, s_, r))
        return transitions

    
    def gather_experience(self, num_episodes=5000):
        """
        Recoge experiencia del entorno y la almacena en el modelo.
        Ejecuta episodios aleatorios para llenar la función de probabilidad empírica.
        """
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            while not done and steps < self.T_max:
                action = self.env.action_space.sample()
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.record_transition(state, action, reward, next_state)
                if reward == -100:  # cliff
                    break
                state = next_state
                steps += 1

    def compute_action_value(self, s, a):
        """
        Devuelve Q(s, a): el valor esperado de tomar la acción a en el estado s.
        """
        q = 0
        # Iteramos las tuplas de la función de probabilidad, que son de la forma: P[s][a] = [(probabilidad, siguiente_estado, recompensa, final)]
        for prob, next_s, reward in self.compute_empirical_p(s, a):
            q += prob * (reward + self.gamma * self.V[next_s])
        return q

    def extract_policy(self):
        """
        Deriva la política óptima a partir de los valores V.
        """
        for s in range(self.n_states):
            # Si el estado es terminal o es parte del cliff, no hacemos nada (no tiene sentido que haya política)
            if (37 <= s <= 46) or (s == 47):
                continue
            state_action_values = [self.compute_action_value(s, a) for a in range(self.n_actions)]
            self.policy[s] = np.argmax(state_action_values)

    def run_episode(self):
        """
        Ejecuta un episodio usando la política actual.
        Termina si se alcanza un estado terminal o el cliff.
        """
        state, _ = self.env.reset()
        done = False
        episode_return = 0
        steps = 0

        while not done and steps < self.T_max:
            action = self.policy[state]
            state, reward, done, truncated, _ = self.env.step(action)
            episode_return += reward
            steps += 1
            if reward == -100:  # Si cae al cliff, termina el episodio
                break

        return episode_return

    def check_policy_performance(self):
        """
        Evalúa la política actual ejecutando episodios reales.
        Devuelve el retorno promedio.
        """
        total_returns = sum(self.run_episode() for _ in range(self.n_test_episodes_per_iteration))
        return total_returns / self.n_test_episodes_per_iteration
    
    def print_state_values(self):
        """
        Imprime el valor actual de cada estado en forma de rejilla 4x12.
        """
        n_rows, n_cols = 4, 12
        print("Valores de los estados:")
        for i in range(n_rows):
            row = ""
            for j in range(n_cols):
                state = i * n_cols + j
                if state < self.n_states:
                    row += f"{self.V[state]:.2f} "
                else:
                    row += "X "
            print(row)
        print("\n")

    def plot_state_values(self, filename="state_values.png"):
        """
        Guarda una imagen 4x12 de los valores de los estados en formato PNG.
        """
        grid = np.zeros((4, 12))
        for s in range(self.n_states):
            row, col = divmod(s, 12)
            if 37 <= s <= 46:
                grid[row, col] = -100
            elif s == 47:
                grid[row, col] = 0
            else:
                grid[row, col] = self.V[s]

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(grid, cmap="coolwarm", vmin=np.min(self.V), vmax=np.max(self.V))

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
        ax.set_title("Valores de los estados")
        # fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def print_policy(self):
        """
        Imprime la política actual en forma de rejilla 4x12 con flechas.
        """
        action_symbols = ['↑', '→', '↓', '←']  # 0: up, 1: right, 2: down, 3: left
        n_rows, n_cols = 4, 12
        print("Política actual:")
        for i in range(n_rows):
            row = ""
            for j in range(n_cols):
                idx = i * n_cols + j
                # Si el estado es parte del cliff ponemos una X, si es el goal, ponemos una G
                if idx == 47:
                    row += "G  "
                    continue
                if 37 <= idx <= 46:
                    row += "X  "
                    continue
                row += f"{action_symbols[self.policy[idx]]}  "
            print(row)
        print()

    def plot_policy(self, filename="policy.png"):
        """
        Guarda una imagen 4x12 con la política óptima en cada estado (flechas).
        """
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
        ax.imshow(np.ones((4,12)), cmap="gray", vmin=0, vmax=1)

        for i in range(4):
            for j in range(12):
                ax.text(j, i, grid[i, j], ha='center', va='center', fontsize=14, color='black')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Política aprendida")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_policy_changes(self, filename="policy_changes.png"):
        """
        Grafica el número de cambios en la política por iteración.
        """

        plt.figure(figsize=(8, 4))
        plt.plot(self.policy_changes_per_iteration)
        plt.title("Cambios en la política por iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Nº de cambios en la política")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_avg_return(self, filename="avg_return.png"):
        """
        Grafica el retorno promedio por iteración.
        """
        plt.figure(figsize=(8, 4))
        plt.plot(self.avg_return_per_iteration)
        plt.title("Retorno promedio por iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Retorno promedio")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


    def run_value_iteration(self, debug=False, performance_check=False, visualize_learning=False, visualize_every=10):
        """
        Ejecuta el algoritmo de iteración de valor hasta convergencia.
        """
        i = 0
        while True:
            delta = 0
            # Para cada estado que no es terminal (es parte del cliff o es el goal)
            for s in range(self.n_states):
                if (37 <= s <= 46) or (s == 47): # Esto no haría falta si la función de probabilidad fuese como la de FrozenLake
                    continue
                # Calculamos el valor de todas las acciones que se pueden realizar desde éste
                state_action_values = [self.compute_action_value(s, a) for a in range(self.n_actions)]
                # Vemos cómo cambia el valor del estado con el de la iteración anterior
                best_action_value = np.max(state_action_values)
                delta = max(delta, abs(best_action_value - self.V[s]))
                # Actualizamos V(s), su valor
                self.V[s] = best_action_value
            # Extraemos la política actual
            prev_policy = self.policy.copy()
            self.extract_policy()
            policy_changes = np.sum(prev_policy != self.policy)
            self.policy_changes_per_iteration.append(policy_changes)
            # Probar la política actual y guardar el retorno promedio
            if performance_check:
                avg_return = self.check_policy_performance()
                self.avg_return_per_iteration.append(avg_return)
            # Imprimir el retorno promedio, el delta, el número de cambios en la política, los valores de los estados y la política
            if debug:
                ret = f"{avg_return:.2f}" if performance_check else "N/A"
                print(f"Iteration {i}: retorno promedio = {ret}, delta = {delta:.5f}, cambios en política = {policy_changes}")
                # Imprimimos los valores de los estados
                self.print_state_values()
                self.print_policy()

            # Visualizar el aprendizaje
            if visualize_learning and i % visualize_every == 0:
                print(f"\n--- Visualizando episodio en iteración {i} ---")
                self.render_episode()

            i += 1
            # Vemos si hemos pasado el umbral de convergencia
            if delta < self.theta:
                break