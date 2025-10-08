import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal

# retuns previous value of U if None is set, a vlue is generated internally
def X(prev_U, a, b):

    if prev_U is None:
        prev_U = np.random.normal(0, 1) 

    current_U = np.random.normal(0, 1)
    result = current_U + (a * prev_U + b) # prev_U == U[n-1]

    return (result, current_U)


def main(i, a, b):

    i = int(i)
    sum = 0

    current_U = None # init condition
    x = []
    for _ in range(i):
        res, current_U = X(current_U, a, b)
        x.append(res)

    return x

if __name__ == "__main__":
    i_arr = []
    y = []

    np.random.seed(69)

    for i in tqdm(np.logspace(0,70,num=71)):
        a = np.random.random_integers(1,10)
        b = np.random.random_integers(1,10)
        i_arr.append(i)
        x = main(5, a, b)

        x_bias_auto_corr = signal.correlate(x,x, mode='full')

        x = np.subtract(x, b) # WRONG #FIXME use -k to unbias not b meanshift
        x_unbias_auto_corr = signal.correlate(x,x, mode='full')

        y.append([x_bias_auto_corr, x_unbias_auto_corr])
        print(f"iter {i}: biased {x_bias_auto_corr} unbiased {x_unbias_auto_corr} A: {a} B: {b}")
    
    # Plot all realisations' correlations on top of each other using stem plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    for idx, (biased, unbiased) in enumerate(y):
        lags_biased = np.arange(-len(biased)//2 + 1, len(biased)//2 + 1)
        lags_unbiased = np.arange(-len(unbiased)//2 + 1, len(unbiased)//2 + 1)
        # Use markerfmt and linefmt to make stems less cluttered
        axs[0].stem(lags_biased, biased, markerfmt='.', linefmt='-', basefmt=" ", label=f'Realisation {idx+1}')
        axs[1].stem(lags_unbiased, unbiased, markerfmt='.', linefmt='-', basefmt=" ", label=f'Realisation {idx+1}')

    axs[0].set_title('Biased Autocorrelation (All Realisations)')
    axs[0].set_xlabel('Lag')
    axs[0].set_ylabel('Correlation')

    axs[1].set_title('Unbiased Autocorrelation (All Realisations)')
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('Correlation')

    # Optionally, add legends if you want to distinguish realisations
    # axs[0].legend()
    # axs[1].legend()

    plt.tight_layout()
    plt.show()
