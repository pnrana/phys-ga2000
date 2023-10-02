# Computational Physics
# Problem Set #4
# 2

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def T(V=None,m = None, a = None,N =20):
    '''A function that computes period(T) for an anharmonic oscillator using Gaussian quadrature

    :param V: function
                potential energy of the particle
    :param m: int or float
                mass of the particle
    :param a: int or float
                amplitude of the oscillator
    :param N: int
                number of points for Gaussian quadrature
    :return T: int or float
                period of the oscillator for  user defined parameters
    '''

    func = lambda x: 1/np.sqrt(V(a)-V(x))
    (gauss_integral, none) = integrate.fixed_quad(func,0,a,n=N)
    return np.sqrt(8 * m) * gauss_integral


def main():
    V = lambda x: x ** 4    #potential of the particle as defined in the problem
    Ts = []                 #list to store amplitudes for a range of a's

    amps = np.arange(0,2.01,0.01)
    for a in amps:
        Ts.append(T(V,1,a))

    plt.plot(amps,Ts)
    plt.title("Periods of Anharmonic Oscillator")
    plt.xlabel("Amplitude (m)")
    plt.ylabel("T (s)")
    #plt.show()
    plt.savefig('problem2.png',dpi=1000)



if __name__ == '__main__':
    main()

