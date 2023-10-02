# Computational Physics
# Problem Set #4
# 3

import numpy as np
from math import factorial
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt

def H(n=None,x=None):
    '''A function to calculate the nth Hermite polynomial recursively
    :param n: int
                nth Hermite polynomial (n = 0 ... inf)
    :param x: int or float
                x for the Hermite polynomial H_n(x)
    :return:
                the nth Hermite polynomial evaluated at x
    '''

    if (n<0):
        raise ValueError("N cannot be negative")
    if (n==0):      #H_0(x) = 1
        return 1
    elif (n==1):    #H_1(x) = 2x
        return (2*x)

    return ((2*x*H(n-1,x)) - (2*(n-1)*H(n-2,x)))


def wave_func(x=None,n= None):
    '''A function to calculate the wavefunction of the nth energy level of the 1D quantum harmonic oscillator
    :param n: int
                nth energy level
    :param x: np.ndarray
               range over which to evaluate the wavefunction
    :return:
                returns the wavefunction for the nth energy level evaluated over a range of x values
    '''

    return(np.exp((-x**2)/2)*H(n,x))/np.sqrt((2**n)*factorial(n)*np.sqrt(np.pi))

def integrand(x,n=5):
    #function that computes <x^2>
    return (x**2)*(wave_func(x,n)**2)

def func_rescale(z=None):
    #function that rescales integral limits
    # -inf....inf ---> -1...+1
    x = z/(1-z**2)
    dx = (1+z**2)/(1-z**2)**2
    return (dx * integrand(x=x))

def gQuad(x,N=100):
    #gaussian quadrature
    (gauss_integral, none) = integrate.fixed_quad(func_rescale,-1,1, n=N)
    return gauss_integral

def gHerm(energy_l,N=100):
    #gauss hermite quadrature
    (x, weight,_) = special.roots_hermite(N,True)
    gHerm_integral = (np.exp(x**2)*integrand(x) * weight).sum()
    return gHerm_integral

def main():

    #a: qho wavefunctions for n=0,..,3 and x=-4,..,4
    x = np.arange(-4,4,0.1)
    wavs = []
    for n in range(0,4):
        wavs.append(wave_func(x,n))
        plt.plot(x,wavs[n],label='n = %s' %n)
    plt.title("Quantum Harmonic Oscillator Wavefunctions")
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    plt.legend()
    #plt.show()
    #plt.savefig('problem3_a.png',dpi=1000)

    plt.clf()       #clear plot

    # b: qho wavefunction for n=30 and x=-10,..,10
    x_values = np.arange(-10,10,0.01)
    wav_b = wave_func(x_values,30)
    plt.plot(x_values,wav_b)
    plt.title("QHO Wavefunction for n=30")
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    #plt.show()
    #plt.savefig('problem3_b.png',dpi=1000)

    #c:
    print("Using Gaussian quadrature:")
    expec_x = gQuad(5)
    print("\t\t<x^2>: ",expec_x)
    print("\t\tUncertainty (RMS position) = ", np.sqrt(expec_x))

    #d:
    print("\nUsing Gauss-Hermite quadrature:")
    expec_xh = gHerm(5)
    print("\t\t<x^2>: ", expec_xh)
    print("\t\tUncertainty (RMS position) = ", np.sqrt(expec_xh))




if __name__ == '__main__':
    main()

