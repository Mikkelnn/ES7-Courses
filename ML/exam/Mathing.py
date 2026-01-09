import numpy as np

Px = [0,2,3,5,3,4]
Py = [2,3,1,4,4,2]
m1 =[3,3]
recError = 0
for n in range(0,6):
    dist = np.sqrt((Px[n]-m1[0])**2+(Py[n]-m1[1])**2)
    print(f"P{n+1} distance to point xq: {dist}")

