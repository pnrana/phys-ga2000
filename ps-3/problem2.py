# Computational Physics
# Problem Set #3
# 3.2

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import UnivariateSpline

#function computes product of matrices by iterating over each element
def naive_multiplication(A,B):
    '''Parameters:
        A: array
            first matrix
        B: array
            second matrix

        computes and returns the product of A & B
        along with the number of operations performed'''
    
    N = len(A)                      #size of matrix
    C = np.zeros([N, N], float)     #array to store the product 
    count = 0                       #count number of operations performed

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
                count += 3          #multiplication, addition, and assignment operation in each iteration

    return C,count


def dot_multiplication(A,B):
    '''Parameters:
        A: array
            first matrix
        B: array
            second matrix

        computes and returns the product of A & B using np.dot()'''
    return(np.dot(A,B))

def main():
    m_sizes = list(range(10,200,20))
    n_counts, dot_counts = [],[]
    n_times , dot_times = [],[]

    #1x1 matrices to calculate time for one np.dot() operation
    #use this to approximate number of operations in subsequent matrix dimensions
    A = np.random.rand(1, 1)
    B = np.random.rand(1, 1)
        
    start_time = time.time()
    for i in range(10):
        dot_multiplication(A,B)
    end_time = time.time()
    
    dot_single_time = (end_time - start_time)/10 #average time of 10 operations


    for N in m_sizes:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        
        #for the naive multiplication function
        start_time = time.time()
        _,count = naive_multiplication(A,B)
        n_counts.append(count)
        n_times.append(time.time()-start_time)

        #for the dot() method
        start_time = time.time()
        dot_multiplication(A,B)
        end_time = time.time()
        dot_counts.append((end_time-start_time)/dot_single_time)    #time for NxN matrices/ time for 1x1 matrices
        dot_times.append(end_time-start_time)


    plt.subplot(121)
    plt.plot(m_sizes,n_counts,label='Naive method')
    plt.plot(m_sizes,dot_counts,linestyle = "dashdot",label = 'dot() method')
    plt.plot(m_sizes,3*np.power(m_sizes,3),linestyle = "dotted", label = '3* N^3')
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Number of operations")
    plt.legend()
    plt.title("Matrix Multiplication Computations")

    
    plt.subplot(122)
    plt.plot(m_sizes, n_times ,label='Naive method')
    plt.plot(m_sizes, dot_times,linestyle= 'dashdot',label='dot() method')
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Time (s)")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Matrix Multiplication Time complexity')

    plt.show()
    #plt.savefig('matrix_multiplication.png')



if __name__ == '__main__':
    main()

