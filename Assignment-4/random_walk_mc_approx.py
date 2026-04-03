import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RandomWalk1000:
    n_states: int = 1000
    max_jump: int = 100
    start_state: int = 500

    def step(self, state, action, rng):
        jump = int(rng.integers(1, self.max_jump + 1))
        next_state = state + action * jump

        if next_state <= 0:
            return 0, -1.0, True
        if next_state > self.n_states:
            return self.n_states + 1, 1.0, True
        return next_state, 0.0, False


def true_value_function(env, tolerance=1e-10, max_iterations=200000):
    values = np.zeros(env.n_states + 2, dtype=float)
    values[0] = -1.0
    values[-1] = 1.0

    for _ in range(max_iterations):
        old_values = values.copy()

        for s in range(1, env.n_states + 1):
            left_sum = 0.0
            right_sum = 0.0

            for jump in range(1, env.max_jump + 1):
                left_state = s - jump
                if left_state <= 0:
                    left_sum += -1.0
                else:
                    left_sum += old_values[left_state]

                right_state = s + jump
                if right_state > env.n_states:
                    right_sum += 1.0
                else:
                    right_sum += old_values[right_state]

            values[s] = 0.5 * (left_sum / env.max_jump) + 0.5 * (right_sum / env.max_jump)

        delta = np.max(np.abs(values - old_values))
        if delta < tolerance:
            break

    return values[1:-1]


def generate_episode(env, rng):
    states = []
    rewards = []
    state = env.start_state

    while True:
        states.append(state)
        action = -1 if rng.random() < 0.5 else 1
        next_state, reward, done = env.step(state, action, rng)
        rewards.append(reward)

        if done:
            break
        state = next_state

    return states, rewards


def run_gradient_mc(env, episodes, alpha, n_features, feature_fn, seed):
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_features, dtype=float)

    for _ in range(episodes):
        states, rewards = generate_episode(env, rng)
        g = 0.0

        for t in range(len(states) - 1, -1, -1):
            g += rewards[t]
            phi = feature_fn(states[t])
            v_hat = float(np.dot(weights, phi))
            weights += alpha * (g - v_hat) * phi

    return weights


def normalize_state(s, env):
    return (s - 1) / (env.n_states - 1)


def make_aggregation_feature_fn(env, groups=10):
    def feature_fn(state):
        phi = np.zeros(groups, dtype=float)
        x = normalize_state(state, env)
        idx = min(int(x * groups), groups - 1)
        phi[idx] = 1.0
        return phi

    return feature_fn, groups


def make_polynomial_feature_fn(env, order=5):
    exponents = np.arange(order + 1)

    def feature_fn(state):
        x = normalize_state(state, env)
        return np.power(x, exponents)

    return feature_fn, order + 1


def make_fourier_feature_fn(env, order=5):
    coeffs = np.arange(order + 1)

    def feature_fn(state):
        x = normalize_state(state, env)
        return np.cos(np.pi * coeffs * x)

    return feature_fn, order + 1


def predict_values(env, weights, feature_fn):
    preds = np.zeros(env.n_states, dtype=float)
    for s in range(1, env.n_states + 1):
        preds[s - 1] = float(np.dot(weights, feature_fn(s)))
    return preds


def plot_approximations(true_values, agg_values, poly_values, fourier_values, output_file):
    states = np.arange(1, len(true_values) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    axes[0].plot(states, true_values, label="True v_pi", linewidth=2, color="black")
    axes[0].plot(states, agg_values, label="State aggregation", linewidth=1.8, color="tab:blue")
    axes[0].set_title("State Aggregation (10 groups)")
    axes[0].set_xlabel("State")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(states, true_values, label="True v_pi", linewidth=2, color="black")
    axes[1].plot(states, poly_values, label="Polynomial (n=5)", linewidth=1.8, color="tab:orange")
    axes[1].set_title("Polynomial Basis (order 5)")
    axes[1].set_xlabel("State")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(states, true_values, label="True v_pi", linewidth=2, color="black")
    axes[2].plot(states, fourier_values, label="Fourier (n=5)", linewidth=1.8, color="tab:green")
    axes[2].set_title("Fourier Basis (order 5)")
    axes[2].set_xlabel("State")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("1000-state Random Walk: Gradient MC Approximation vs True v_pi")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Gradient MC on 1000-state random walk")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes per method")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--output", type=str, default="random_walk_value_approx.png", help="Plot output file")
    args = parser.parse_args()

    env = RandomWalk1000()
    true_values = true_value_function(env)

    agg_feature_fn, agg_dim = make_aggregation_feature_fn(env, groups=10)
    poly_feature_fn, poly_dim = make_polynomial_feature_fn(env, order=5)
    fourier_feature_fn, fourier_dim = make_fourier_feature_fn(env, order=5)

    agg_w = run_gradient_mc(
        env=env,
        episodes=args.episodes,
        alpha=2e-5,
        n_features=agg_dim,
        feature_fn=agg_feature_fn,
        seed=args.seed,
    )
    poly_w = run_gradient_mc(
        env=env,
        episodes=args.episodes,
        alpha=1e-4,
        n_features=poly_dim,
        feature_fn=poly_feature_fn,
        seed=args.seed + 1,
    )
    fourier_w = run_gradient_mc(
        env=env,
        episodes=args.episodes,
        alpha=5e-5,
        n_features=fourier_dim,
        feature_fn=fourier_feature_fn,
        seed=args.seed + 2,
    )

    agg_values = predict_values(env, agg_w, agg_feature_fn)
    poly_values = predict_values(env, poly_w, poly_feature_fn)
    fourier_values = predict_values(env, fourier_w, fourier_feature_fn)

    plot_approximations(true_values, agg_values, poly_values, fourier_values, args.output)

    print(f"Saved plot: {args.output}")
    print(f"RMSE state aggregation: {rmse(agg_values, true_values):.6f}")
    print(f"RMSE polynomial n=5: {rmse(poly_values, true_values):.6f}")
    print(f"RMSE Fourier n=5: {rmse(fourier_values, true_values):.6f}")


if __name__ == "__main__":
    main()
