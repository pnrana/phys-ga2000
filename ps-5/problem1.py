#ps5
#problem 1
#Gamma function

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def integrand(x,a):
    '''
    Function that evaluates the integrand as a function of x
    :param x: np.float64 or np.float32
                x value
    :param a: int
                a value
    :return: np.float64 or np.float32
                integrand value
    '''

    return (x**(a-1)*np.exp(-x))

def func_rescale(z=None,a=2):
    '''
    Function that rescales the integral limits that puts the peak in the middle of the integration range
    :param z: np.float64 or np.float32
                takes in x values to be rescaled
    :param a: int
                a value for the gamma function
    :return:
                integrand with rescaled limits
    '''

    x = ((a-1)*z)/(1-z)
    dx = (a-1)/((1-z)**2)
    return (dx * integrandest(x,a))

def integrandest(x,a):
    #alternative expression for the integrand of Gamma function to avoid potential numerical errors
    return np.exp(((a-1)*np.log(x))- x)

def gamma(a,N=50):
    '''Function that evaluates the gamma function for particular x and a values using Gaussian quadrature
    :param x: np.float64 or np.float32
                x value
    :param a: int
                a value
    :param N: int
                number of points to evaluate the integral
    :return: np.float64 or np.float32
                gamma function value
    '''

    (gauss_integral, _) = integrate.fixed_quad(func_rescale, 0, 1 ,args=(a,) ,n=N)
    return gauss_integral

def main():

    #a) plot integrand
    x= np.linspace(0,5,100)

    for a in range(2,5):
        gam = integrand(x,a)
        plt.plot(x,gam,label='a=%s'%a)

    plt.title("Integrand of the Gamma function as a function of x")
    plt.xlabel("x")
    plt.ylabel("integrand f(x)")
    plt.legend()
    #plt.show()
    #plt.savefig("ps5p1a.png",dpi=900)

    #c) rescaling integral limits - > function func_rescale

    #e) test gamma function for a=3/2
    print('1/2 * (pi)^(1/2) =', 0.5 * np.sqrt(np.pi))
    print('\u0393(3/2) =',gamma(3/2),'\n')
    print("Difference between the two values:", 0.5 * np.sqrt(np.pi) - gamma(3/2),'\n')

    #f) evaluate gamma function for a=3,6,10
    for a in (3,6,10):
        print('\u0393(%s) ='%a, gamma(a))


if __name__ == '__main__':
    main()

