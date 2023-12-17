import numpy as np
# Step 1: Simulate the Unemployment Rate (Ornstein-Uhlenbeck Process)

def simulate_ou_process(theta, mu, sigma, dt, X0, N):
    """
    Simulate the Ornstein-Uhlenbeck process for unemployment rate.
    theta: rate of reversion to mean
    mu: long-term mean
    sigma: volatility
    dt: time step
    X0: initial value
    N: number of steps
    """
    X = np.zeros(N)
    X[0] = X0
    for t in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
    return X

def simulate_gbm_process(mu, sigma, S0, dt, N):
    """
    Geometric Brownian Motion process simulation for variables with drift and volatility.
    """
    S = np.zeros(N)
    S[0] = S0
    for t in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    return S

def simulate_jump_diffusion_process(lambda_, mu, sigma, delta, dt, S0, N):
    """
    Merton Jump-Diffusion process simulation for variables that can have sudden jumps.
    """
    S = np.zeros(N)
    S[0] = S0
    for t in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        J = np.random.poisson(lambda_ * dt)
        jump_sum = sum(np.random.normal(mu, delta) for _ in range(J)) if J > 0 else 0
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW) + jump_sum
    return S