# Computational Physics
# Problem Set #4
# 1

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def integrand(x = None):
    #function to perform the integration on
    return (x ** 4) * np.exp(x) / ((np.exp(x) - 1) ** 2)

def cv(T = None,V = 1000e-6,rho = 6.022e28,thetad = 428,N = 50):
    '''A function that computes the Cv for a T

    :param T: float
                Temperature in K
    :param V: float
                volume of the solid in m^3 (set to 1000 cm^3 or 0.001 m^3 for solid aluminum)
    :param rho: float
                density in m^-3 (set to 6.022e28 m^-3 for aluminum)
    :param thetad: float
                Debye temperature in K (set to 428K for aluminum)
    :param N: int
                number of sample points for gaussian quadrature
    :return Cv: float
                heat capacity calculated for user defined parameters
    '''

    kb = 1.380649e-23  #Boltzmann constant in Joules per Kelvin
    coef = 9*V*rho*kb*np.power((T/thetad),3)

    (gauss_integral, none) = integrate.fixed_quad(integrand, 0, (thetad/T), n=N)
    return(coef*gauss_integral)

def main():

    #b) plotting Cv for T = [5,500] with N=50
    temps = np.arange(5,500,0.1)
    heatcaps = []

    for T in temps:
        heatcaps.append(cv(T))

    plt.plot(temps,heatcaps,color="red")
    plt.title("Heat Capacity of Aluminium")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$C_V$ ($J.K^{-1}$)")
    #plt.show()
    plt.savefig("ps4p1.png",dpi=1000)

    #clear the plot
    plt.clf()

    #c) testing convergence
    N = np.arange(10,71,10)
    cv_ns = []      #each element is a list of cv values for a particular n

    for n in N:
        cvs = []
        for T in temps:
            cvs.append(cv(T,N=n))
        cv_ns.append(cvs)
        plt.plot(temps,cvs,label="n =%s" %n)

    plt.title("Cv vs T for different N")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$C_V$ ($J.K^{-1}$)")
    plt.legend()
    #plt.show()
    plt.savefig("ps4p1b.png",dpi=1000)  #this scale isn't really useful because the curves overlap so you can't see their difference

    #clear plot
    plt.clf()

    #calculate the difference of Cvs obtained from each n compared to n=70
    cv_ns = np.array(cv_ns)
    for i in range(len(N)-1):
        plt.plot(np.log10(temps), np.log10(np.abs((cv_ns[i] - cv_ns[-1]) / cv_ns[-1])),
                 label=f"(n$_{(len(N))}$$_0$-n$_{i+1}$$_0$)/n$_{len(N)}$$_0$")

    plt.title("Log of relative difference in Cv with reference to N=70")
    plt.xlabel("log$_{10}$ Temperature (K)")
    plt.ylabel("log$_{10}$ (Relative difference of Cv) ")
    plt.legend()
    #plt.show()
    plt.savefig("ps4p1c.png",dpi=1000)



if __name__ == '__main__':
    main()

