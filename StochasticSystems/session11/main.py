import scipy.fft
import scipy.io as io
import numpy as np
import pathlib as path
import matplotlib.pyplot as plt 
from scipy import signal
import scipy.fft as fft
import math

file = path.Path("./hourlyDataTrafficInBits.mat")

mat = io.loadmat(file)["hourlyDataTrafficInBits"][0]

def plot_data(data, title: str, x_label: str, y_label: str, scale_x = None, center_x_axis: bool = False):
    num = np.random.randint(0, 1000)
    
    if center_x_axis:
        half = len(data) // 2
        # print(half)
        x = np.arange(-half, -half + len(data))
    else:
        x = np.arange(len(data))
    
    if scale_x is not None:
        x = scale_x(x)

    plt.figure(num)
    plt.plot(x, data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # ax.set_xlim([xmin, xmax])
    # plt.show()

def correlate(x,y):
    return signal.correlate(x,y, mode='full')

def fft_data(data):
    return fft.ifftshift(fft.fft(x=data))

def periodogram_data(acf):
    return fft.ifftshift(fft.fft(x=acf))

def difference(data, T):
    data_diff = []
    for i in range(len(data)):
        data_diff.append(data[i] - data[i - (int(T)%len(data))])

    return data_diff

def main():
    scalar = lambda x: x * ((1/3600) / len(mat_zero_mean_fft)) #((len(mat_zero_mean_fft) - 1) / len(mat_zero_mean_fft))
    
    # remove mean from data
    mat_zero_mean = mat - np.mean(mat)

    autocorr = correlate(mat_zero_mean,mat_zero_mean)
    
    periodogram = periodogram_data(autocorr)

    mat_zero_mean_fft = fft_data(mat_zero_mean)

    fft_max_frequency = 24

    mat_zero_mean_diff = difference(mat_zero_mean, fft_max_frequency)

    plot_data(mat, "Hourly trafic", "Hour", "bits")
    plot_data(mat_zero_mean, "Zero mean hourly trafic", "Hour", "bits")
    plot_data(autocorr, "Autocorr", "lag", "?", center_x_axis=True)
    plot_data(periodogram, "Periodogram", "?", "?", scale_x=scalar) #TODO fix axis scaling
    plot_data(mat_zero_mean_fft, "FFT", "Hours^-1", "?", scale_x=scalar,  center_x_axis=True) #TODO fix axis scaling
    plot_data(mat_zero_mean_diff, "Difference", "Hour", "bits")

    plt.show()

if __name__ == "__main__":
    main()
