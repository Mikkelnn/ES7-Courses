import numpy as np
import scipy.signal as ss

SAMPLES = 128

def loadData():
    return np.loadtxt('Opgave.dat.txt')

def getMatrix(autocorrArr):
    length = len(autocorrArr)
    RMatrix = np.zeros((length, length))

    for n in range(length):
        for m in range(length):
            RMatrix[n,m] = autocorrArr[np.abs(n-m)]
    
    return RMatrix

def LPC(data):
    window = ss.windows.hamming(SAMPLES)

    windowedData = data[0:SAMPLES] * window

    autocorr = ss.correlate(windowedData, windowedData, mode='full')[SAMPLES-1:] # only get from lag 0...N

    matrix = getMatrix(autocorr)
    
    print(matrix.shape)
    # print(len(autocorr))

    return matrix

def main():
    data = loadData()
    autocorr = LPC(data)
    print(autocorr)


if __name__ == "__main__":
    main()