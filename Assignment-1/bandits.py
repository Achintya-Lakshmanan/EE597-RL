import matplotlib.pyplot as plt
import numpy as np

class GaussianBandit:
    def __init__(self, mu, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.N = 0
        self.q_estimate = 0.0 
        self.sum_rewards = 0.0
    
    def pull(self):
        # Rewards are N(mu, 1)
        return np.random.normal(self.mu, self.sigma)
    
    def update(self, x):
        self.N += 1
        self.sum_rewards += x
        self.q_estimate += (x - self.q_estimate) / self.N

def run_explore_first(means, n_steps=1000, explore_blocks=50):
    """
    Explore First: Explore each arm for `explore_blocks` times, then exploit.
    """
    bandits = [GaussianBandit(mu) for mu in means]
    n_arms = len(bandits)
    optimal_mu = np.max(means)
    regrets = np.zeros(n_steps)
    
    step = 0
    # Exploration
    for _ in range(explore_blocks):
        for j in range(n_arms):
            if step >= n_steps: break
            
            x = bandits[j].pull()
            bandits[j].update(x)
            
            regrets[step] = optimal_mu - bandits[j].mu
            step += 1
            
    # Exploitation
    while step < n_steps:
        j = np.argmax([b.q_estimate for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        regrets[step] = optimal_mu - bandits[j].mu
        step += 1
        
    return np.cumsum(regrets)

def run_epsilon_greedy(means, n_steps=1000, epsilon=0.1):
    bandits = [GaussianBandit(mu) for mu in means]
    n_arms = len(bandits)
    optimal_mu = np.max(means)
    regrets = np.zeros(n_steps)
    
    for i in range(n_steps):
        if np.random.random() < epsilon:
            j = np.random.randint(n_arms)
        else:
            j = np.argmax([b.q_estimate for b in bandits])
            
        x = bandits[j].pull()
        bandits[j].update(x)
        
        regrets[i] = optimal_mu - bandits[j].mu
        
    return np.cumsum(regrets)

def run_ucb(means, n_steps=1000, c=2):
    bandits = [GaussianBandit(mu) for mu in means]
    n_arms = len(bandits)
    optimal_mu = np.max(means)
    regrets = np.zeros(n_steps)
    
    for i in range(n_steps):
        ucb_values = []
        for b in bandits:
            if b.N == 0:
                ucb_values.append(float('inf'))
            else:
                ucb_values.append(b.q_estimate + c * np.sqrt(np.log(i + 1) / b.N))
        
        j = np.argmax(ucb_values)
        x = bandits[j].pull()
        bandits[j].update(x)
        
        regrets[i] = optimal_mu - bandits[j].mu
        
    return np.cumsum(regrets)

def run_thompson_sampling(means, n_steps=1000):
    bandits = [GaussianBandit(mu) for mu in means]
    n_arms = len(bandits)
    optimal_mu = np.max(means)
    regrets = np.zeros(n_steps)
    
    # Bayesian Prior: N(0, 1)
    # Likelihood: N(mu, 1)
    
    for i in range(n_steps):
        samples = []
        for b in bandits:
            # Posterior mean = (Sum of rewards) / (N + 1)
            # Posterior variance = 1 / (N + 1) using prior precision=1 and likelihood precision=1
            
            # Using formulas for Gaussian with known sigma=1 and N(0,1) prior
            precision_0 = 1.0 # 1/sigma_prior^2
            precision_data = 1.0 # 1/sigma^2
            
            precision_n = precision_0 + b.N * precision_data
            sigma_sq_n = 1.0 / precision_n
            
            mu_n = sigma_sq_n * (0 + b.sum_rewards * precision_data) # mean_0 = 0
            
            sample = np.random.normal(mu_n, np.sqrt(sigma_sq_n))
            samples.append(sample)
        
        j = np.argmax(samples)
        x = bandits[j].pull()
        bandits[j].update(x)
        
        regrets[i] = optimal_mu - bandits[j].mu
        
    return np.cumsum(regrets)

def main():
    n_experiments = 500
    n_steps = 1000
    n_arms = 10
    
    regret_explore_first = np.zeros(n_steps)
    regret_epsilon_greedy = np.zeros(n_steps)
    regret_ucb = np.zeros(n_steps)
    regret_thompson = np.zeros(n_steps)
    
    print(f"Running {n_experiments} experiments...")
    
    for _ in range(n_experiments):
        # Draw mu_i randomly from N(0, 1)
        means = np.random.normal(0, 1, n_arms)
        
        regret_explore_first += run_explore_first(means, n_steps, explore_blocks=5) # 5*10 = 50 steps exploration
        regret_epsilon_greedy += run_epsilon_greedy(means, n_steps, epsilon=0.1)
        regret_ucb += run_ucb(means, n_steps, c=1) # c=1 or c=sqrt(2) often used
        regret_thompson += run_thompson_sampling(means, n_steps)
        
    regret_explore_first /= n_experiments
    regret_epsilon_greedy /= n_experiments
    regret_ucb /= n_experiments
    regret_thompson /= n_experiments
    
    plt.figure(figsize=(10, 6))
    plt.plot(regret_explore_first, label='Explore First')
    plt.plot(regret_epsilon_greedy, label='Epsilon Greedy (0.1)')
    plt.plot(regret_ucb, label='UCB')
    plt.plot(regret_thompson, label='Thompson Sampling')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Average Cumulative Regret')
    plt.title(f'Bandit Algorithms Comparison ({n_experiments} runs)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()