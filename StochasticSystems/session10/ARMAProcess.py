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
    a = np.random(0,2)
    b = np.random(-0.5,2)
    x = []
    x[0] = 0

    for var in tqdm(range(1, 5)):
        Zprev = np.random.normal(0, var)
        for a in np.linspace(0, 2, num=4):
            for b in np.linspace(-2, 2, num=8):
                for n in range(1,100):
                    Znow = np.random.normal(0, var)
                    x[n]=a*x[n-1] + b*Zprev + Znow
                    Zprev = Znow
            

    plt.stem(y) #Error plot
    plt.show()