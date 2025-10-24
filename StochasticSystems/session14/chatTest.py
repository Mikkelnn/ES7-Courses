import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 10      # large n makes the curve smoother
p = 0.3      # probability of success

# Range of possible outcomes
k = np.arange(0, n + 1)
x = k / n    # normalized outcome (fraction of successes)

# PMF values
pmf = binom.pmf(k, n, p)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, pmf, linewidth=2)
plt.title(f'Normalized Binomial PMF (n={n}, p={p})', fontsize=14)
plt.xlabel('Fraction of Successes (k/n)', fontsize=12)
plt.ylabel('Probability Mass', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
