#Newman 8.3
#The lorenz equations

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#diff eqs
def dfdt(t,xyz,sig=10.0,r=28.0,b=8.0/3.0):
    '''

    :param xyz: np.array of (x,y,z)
                stores the current values of x,y,z
    :param sig: float
                sigma
    :param r:  float
    :param b: float
    :return np.array of (f_x,f_y,f_z):
            current values of dx/dt,dy/dt,dz/dt
    '''
    #current state
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    #differential equations
    fx = sig*(y-x)
    fy = r*x - y - x*z
    fz = x*y - b*z

    return np.array([fx,fy,fz],float)


def main():
    #constants
    sig = 10.0
    r = 28.0
    b = 8.0/3.0

    #time range
    tmin = 0.0
    tmax = 50.0


    #initial conditions
    xyz=np.array([0.0,1.0,0.0],float)


    #solving using RK45
    results = integrate.solve_ivp(dfdt, [tmin, tmax], xyz,method='RK45',max_step=0.01)
    print(results)

    
    plt.plot(results.t,results.y[1])
    plt.xlabel("time (s)")
    plt.ylabel("y")
    plt.title("Lorenz equations")
    #plt.savefig("ps8_2_yvst.png",dpi=1000)
    plt.show()
    plt.clf()


    plt.plot(results.y[0],results.y[2])
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Lorenz equations")
    #plt.savefig("ps8_2_xvsz.png",dpi=1000)
    plt.show()


if __name__ == '__main__':
    main()



