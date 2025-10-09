import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# retuns previous value of U if None is set, a vlue is generated internally
def X(prev_U):

    if prev_U is None:
        prev_U = np.random.normal(3, 9) 

    current_U = np.random.normal(3, 9)
    result = current_U - ((1/2) * prev_U) # prev_U == U[n-1]

    return (result, current_U)

def Y(A):
    return A + np.random.normal(0, 1)


def main(i):

    i = int(i)
    sum = 0

    #current_U = None # init condition
    #for _ in range(i):
    #    res, current_U = X(current_U)
    #    sum = sum + res
    
    A = np.random.uniform(0, 3)
    for _ in range(i):
        res = Y(A)
        sum = sum + res

    mean = (1/i)*sum

    return  (mean, A)

if __name__ == "__main__":
    a = 1
    b = 0
    x = []
    y = []

    for i in tqdm(np.logspace(0,7,num=8)):
        x.append(i)
        (mean, A) = main(i)
        y.append(mean-A)
        print(f"iter {i}: mean {mean} A: {A}")

    plt.stem(y) #Error plot
    plt.show()