import numpy as np
import matplotlib.pyplot as vini

seed = 123456 #np.random.randint(1000000)

print(seed)
rng = np.random.default_rng(seed)

def realise_gauss(mean, variance):
    return rng.normal(loc=mean, scale=np.sqrt(variance))

def plot(x, y, h, sigma_w, sigma_z):
    n = np.arange(len(x))  # indices for samples

    # Create a figure with 2 rows and 1 column of subplots
    fig, axes = vini.subplots(2, 1, figsize=(8, 6))

    # First plot — X[n]
    axes[0].stem(n, x)
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('X[n]')
    axes[0].set_title(f'Stem Plot of X[n], h = {h}, sigma_w = {sigma_w}, sigma_z = {sigma_z}')

    # Second plot — Y[n]
    axes[1].stem(n, y)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Y[n]')
    axes[1].set_title(f'Stem Plot of Y[n], h = {h}, sigma_w = {sigma_w}, sigma_z = {sigma_z}')

    # Adjust spacing between plots so titles/labels don’t overlap
    vini.tight_layout()

    # Show both plots in the same window
    vini.show()

def lmmse_scalar(data, theta):
    mean_data = np.mean(data)
    mean_theta = np.mean(theta)
    cov_thetay = np.mean((theta - mean_theta) * (data - mean_data))
    var_y = np.var(data)
    return mean_theta + cov_thetay / var_y * (data - mean_data)

def main():
    h = 0.95 # Must not be greater than one, since no negative variance
    sigma_w = 1
    sigma_z = 0.1
    sigma_y = sigma_z/(1-h**2)

    estimate = [0]
    y_n = []
    x_n = []
    y_0 = realise_gauss(0, sigma_y)
    y_n.append(y_0)
    x_n.append(0)

    N = 10
    for n in range(1, N):
        w_n = realise_gauss(0, sigma_w)
        z_n = realise_gauss(0, sigma_z)
        y_n.append(h * y_n[n-1] + z_n)
        x_n.append(y_n[n] + w_n) 

        estimate.append(lmmse_scalar(data=x_n, theta=estimate))
        print(estimate)

    plot(x_n, y_n, h, sigma_w, sigma_z)

if __name__ == "__main__":
    main()