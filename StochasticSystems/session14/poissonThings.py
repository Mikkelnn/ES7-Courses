import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

rng = np.random.default_rng(seed=42)

def realise_poisson(k: int, mu: float, loc):
    
    realised = poisson.pmf(k=k, mu=mu, loc=loc)

    return realised

def realise_poisson_k(k: int, mu: float, loc):
    
    realised = poisson.pmf(k=k, mu=mu/k, loc=loc)

    return realised

def realise_binomial_n(n: int, p: float, k) -> list:

    realised = binom.pmf(k=k, n=n, p=p/n)

    return realised

def realise_binomial(n: int, p: float, k):

    realised = binom.pmf(k=k, n=n, p=p)

    return realised

def plot_pmf(x, y, samps, title = "No title"):
    plt.figure(np.random.random_integers(0, 1000))
    plt.plot(x, y)
    plt.hist(samps)
    plt.title(title)

def plot_hist(series, title = "No title"):
    plt.figure(np.random.random_integers(0, 1000))
    plt.hist(series)
    plt.title(title)

def show_all_figures():
    plt.show()

def main():
    n = 10
    k = np.arange(0, n+1)

    bino = realise_binomial(n=n, p=0.3, k=k)
    bino_samps = binom.rvs(n=n, p=0.3, size=300000)

    plot_pmf(x=k, y=bino*300000, samps=bino_samps, title="Binomial Pmf of p=0.3")

    mu = 3
    nnn = 10
    kkk = np.arange(0, nnn+1)
    loc = 0

    pois = realise_poisson(k=kkk, mu=mu, loc=loc)
    pois_samps = poisson.rvs(mu, size=300000)

    plot_pmf(x=kkk, y=pois*300000, samps=pois_samps, title=f"Poisson Pmf of mu={mu}")

    # Create a single plot comparing binomial and Poisson distributions
    plt.figure(figsize=(12, 8))
    
    for n_val in [10, 100, 1000]:
        k_vals = np.arange(0, min(n_val+1, 50))  # Limit range for large n values
        
        # Binomial distribution
        bino_vals = realise_binomial_n(n=n_val, p=3, k=k_vals)
        plt.plot(k_vals, bino_vals, label=f"Binomial n={n_val}, p=3/n", alpha=0.7)
        
        # Poisson distribution  
    mu = 3
    pois_vals = poisson.pmf(k=k_vals, mu=mu)
    plt.plot(k_vals, pois_vals, '--', label=f"Poisson μ=3, n={n_val}", alpha=0.7)
    
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.title('Comparison of Binomial and Poisson Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    show_all_figures()

if __name__ == "__main__":
    main()