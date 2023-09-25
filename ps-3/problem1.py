# Computational Physics
# Problem Set #3
# 3.1

import matplotlib.pyplot as plt
import numpy as np

#function defined in the problem that takes a number and returns the resulting number 
def f(x):                                   
    return (x*(x-1))

#calculate  derivative using fundamental definition of the derivative
def dif_f(f=None,x=None,dx=None):
    '''
    Params:
        f : function
        x : int or float
        dx: int or float

    Returns derivative of the function f at x with del set to dx
    '''
        
    return ((f(x+dx)-f(x))/dx)              

def main():
    x = 1
    dx = 1.e-2
    print("dx =",dx,"---> df/dx=",dif_f(f,x,dx),'\n\t\tDifference from true df/dx = ', dif_f(f,x,dx)-1)

    dels = [1.e-4,1.e-6,1.e-8,1.e-10,1.e-12,1.e-14]
    diffs = []
    divs = []
    for dx in dels:
        print("\ndx =",dx,"---> df/dx=",dif_f(f,x,dx),'\n\t\tDifference from true df/dx = ', dif_f(f,x,dx)-1)
        
        


if __name__ == '__main__':
    main()

