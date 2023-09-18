import numpy as np
import matplotlib.pyplot as plt
import timeit

def main():
    print("\nCalculating Madelung Constant using a loop:")
    loopy = '''
L=100
V=np.float32(0)

for i in range(-L,L+1):
    for j in range(-L,L+1):
        for k in range(-L,L+1):
            if (i==j==k==0):
                continue
            if (i+j+k)%2: 
                #print(-(i*i + j*j + k*k)**(-1/2))
                V-=(i*i + j*j + k*k)**(-1/2)
            else:
                #print((i*i + j*j + k*k)**(-1/2))
                V+=(i*i + j*j + k*k)**(-1/2)


print("\tL = ",V)
    '''
    print("\tTime = ",timeit.timeit(loopy,number=1,globals=globals())," seconds")

    print("\nCalculating Madelung Constant using a 2D array:")
    vectorized='''
L=100
ijk = np.vstack(np.mgrid[-L:L+1,-L:L+1,-L:L+1]).reshape(3,-1).T

# Find the row indices where the row is not equal to [0, 0, 0]
non_zero_indices = np.any(ijk != [0, 0, 0], axis=1)

# Filter the rows using the non_zero_indices
ijk = ijk[non_zero_indices]

sign = np.ones(len(ijk))
sign [ijk.sum(axis=1)%2 ==1] = -1

V = sign*((ijk[:,0]**2 + ijk[:,1]**2 +ijk[:,2]**2)**(-1/2))
print("\tL = ",V.sum())

    '''
    
    

    print("\tTime = ",timeit.timeit(vectorized,number=1,globals=globals())," seconds")


if __name__ == "__main__":
    main()
