import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CliffWalkingEnv:
    rows: int = 4
    cols: int = 12

    action_reward = -1
    cliff_reward = -100
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    def __post_init__(self):
        self.start = (self.rows - 1, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.n_states = self.rows * self.cols
        self.n_actions = 4  # up, right, down, left

    def to_state(self, pos):
        r, c = pos
        return r * self.cols + c

    def from_state(self, s):
        return divmod(s, self.cols)

    def reset(self):
        return self.to_state(self.start)

    def step(self, state, action):
        r, c = self.from_state(state)

        if action == 0:  # up
            r = max(0, r - 1)
        elif action == 1:  # right
            c = min(self.cols - 1, c + 1)
        elif action == 2:  # down
            r = min(self.rows - 1, r + 1)
        elif action == 3:  # left
            c = max(0, c - 1)

        next_pos = (r, c)

        if next_pos[0] == self.rows - 1 and 1 <= next_pos[1] <= self.cols - 2:
            # Cliff: large penalty and reset to start.
            reward = self.cliff_reward
            next_pos = self.start
            done = False
        elif next_pos == self.goal:
            reward = self.action_reward
            done = True
        else:
            reward = self.action_reward
            done = False

        return self.to_state(next_pos), reward, done


def epsilon_greedy_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(q_table.shape[1])
    return int(np.argmax(q_table[state]))


def run_sarsa(env, episodes, alpha, gamma, epsilon, max_steps, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    episode_rewards = np.zeros(episodes, dtype=float)

    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy_action(q, state, epsilon, rng)
        total_reward = 0.0

        for _ in range(max_steps):
            next_state, reward, done = env.step(state, action)
            next_action = epsilon_greedy_action(q, next_state, epsilon, rng)

            td_target = reward + gamma * q[next_state, next_action] * (0.0 if done else 1.0)
            q[state, action] += alpha * (td_target - q[state, action])

            total_reward += reward
            state, action = next_state, next_action

            if done:
                break

        episode_rewards[ep] = total_reward

    return episode_rewards, q


def run_q_learning(env, episodes, alpha, gamma, epsilon, max_steps, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    episode_rewards = np.zeros(episodes, dtype=float)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = epsilon_greedy_action(q, state, epsilon, rng)
            next_state, reward, done = env.step(state, action)

            td_target = reward + gamma * np.max(q[next_state]) * (0.0 if done else 1.0)
            q[state, action] += alpha * (td_target - q[state, action])

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards[ep] = total_reward

    return episode_rewards, q


def greedy_policy_from_q(env, q_values):
    return np.argmax(q_values, axis=1)


def policy_to_grid_strings(env, policy):
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    grid = np.empty((env.rows, env.cols), dtype=object)

    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            if pos == env.start:
                grid[r, c] = "S"
            elif pos == env.goal:
                grid[r, c] = "G"
            elif r == env.rows - 1 and 1 <= c <= env.cols - 2:
                grid[r, c] = "C"
            else:
                s = env.to_state(pos)
                grid[r, c] = arrows[int(policy[s])]

    return grid


def trace_greedy_path(env, policy, max_steps=100):
    state = env.reset()
    path = [env.from_state(state)]

    for _ in range(max_steps):
        action = int(policy[state])
        next_state, _, done = env.step(state, action)
        path.append(env.from_state(next_state))
        state = next_state
        if done:
            break

    return path


def plot_policy_diagram(env, policy, title, output_file):
    grid = policy_to_grid_strings(env, policy)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, linewidth=0.8, alpha=0.5)

    for r in range(env.rows):
        for c in range(env.cols):
            txt = grid[r, c]
            if txt == "C":
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="tomato", alpha=0.35)
                )
            elif txt == "S":
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="skyblue", alpha=0.35)
                )
            elif txt == "G":
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgreen", alpha=0.35)
                )
            ax.text(c, r, txt, ha="center", va="center", fontsize=13, fontweight="bold")

    path = trace_greedy_path(env, policy)
    if len(path) > 1:
        xs = [c for (_, c) in path]
        ys = [r for (r, _) in path]
        ax.plot(xs, ys, color="black", linewidth=2, marker="o", markersize=3, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close(fig)


def policy_is_cliff_hugging(env, policy):
    # Check if policy mostly moves right on the row directly above the cliff.
    row_above_cliff = env.rows - 2
    right_moves = 0
    checked = 0

    for c in range(1, env.cols - 2):
        s = env.to_state((row_above_cliff, c))
        if int(policy[s]) == 1:
            right_moves += 1
        checked += 1

    return right_moves >= max(1, int(0.6 * checked))


def main():
    parser = argparse.ArgumentParser(description="Compare SARSA and Q-learning on Cliff Walking")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--runs", type=int, default=1000, help="Independent runs to average")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument(
        "--output",
        type=str,
        default="cliff_walking_episode_rewards.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    env = CliffWalkingEnv()

    sarsa_rewards = np.zeros((args.runs, args.episodes), dtype=float)
    q_learning_rewards = np.zeros((args.runs, args.episodes), dtype=float)
    sarsa_q_sum = np.zeros((env.n_states, env.n_actions), dtype=float)
    q_learning_q_sum = np.zeros((env.n_states, env.n_actions), dtype=float)

    for run in range(args.runs):
        seed = 2026 + run
        sarsa_ep_rewards, sarsa_q = run_sarsa(
            env=env,
            episodes=args.episodes,
            alpha=env.alpha,
            gamma=env.gamma,
            epsilon=env.epsilon,
            max_steps=args.max_steps,
            seed=seed,
        )
        q_ep_rewards, q_q = run_q_learning(
            env=env,
            episodes=args.episodes,
            alpha=env.alpha,
            gamma=env.gamma,
            epsilon=env.epsilon,
            max_steps=args.max_steps,
            seed=seed,
        )
        sarsa_rewards[run] = sarsa_ep_rewards
        q_learning_rewards[run] = q_ep_rewards
        sarsa_q_sum += sarsa_q
        q_learning_q_sum += q_q

    mean_sarsa = sarsa_rewards.mean(axis=0)
    mean_q = q_learning_rewards.mean(axis=0)
    mean_sarsa_q = sarsa_q_sum / args.runs
    mean_q_learning_q = q_learning_q_sum / args.runs

    sarsa_policy = greedy_policy_from_q(env, mean_sarsa_q)
    q_learning_policy = greedy_policy_from_q(env, mean_q_learning_q)

    episodes_axis = np.arange(1, args.episodes + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_axis, mean_sarsa, label="SARSA", linewidth=2)
    plt.plot(episodes_axis, mean_q, label="Q-learning", linewidth=2)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Reward per Episode")
    plt.title("Cliff Walking: Reward per Episode")
    plt.ylim(bottom=-100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    sarsa_policy_file = "sarsa_policy_diagram.png"
    q_policy_file = "q_learning_policy_diagram.png"
    plot_policy_diagram(env, sarsa_policy, "SARSA Greedy Policy", sarsa_policy_file)
    plot_policy_diagram(env, q_learning_policy, "Q-learning Greedy Policy", q_policy_file)

    sarsa_hugs = policy_is_cliff_hugging(env, sarsa_policy)
    q_hugs = policy_is_cliff_hugging(env, q_learning_policy)

    print(f"Plot saved to: {args.output}")
    print(f"SARSA policy diagram saved to: {sarsa_policy_file}")
    print(f"Q-learning policy diagram saved to: {q_policy_file}")
    print(f"Final episode reward (SARSA): {mean_sarsa[-1]:.2f}")
    print(f"Final episode reward (Q-learning): {mean_q[-1]:.2f}")


if __name__ == "__main__":
    main()
