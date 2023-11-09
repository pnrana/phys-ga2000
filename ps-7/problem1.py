#ps7
#Brent

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# parabolic step function from lecture notes
def parabolic_step(func=None, a=None, b=None, c=None):
    """returns the minimum of the function as approximated by a parabola"""
    fa = func(a)
    fb = func(b)
    fc = func(c)
    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)
    # If singular, just return b
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)


#my implementation of brent's method
def brent(func=None,astart=None,cstart=None,tol=1.e-5,maxiter=1000):
    '''
    :param func: function to minimize
    :param astart: bracketing interval start
    :param bstart: bracketing interval end
    :param tol:     tolerance
    :param maxiter: maximum number of iterations
    :return: minimum, value at minimum, number of iterations
    '''

    #plotting for visualization purposes
    xgrid = np.linspace(0,0.5,10000)
    plt.plot(xgrid, func(xgrid))

    #golden ratio
    gratio = (3. - np.sqrt(5)) / 2

    #initial values
    a = astart
    c = cstart
    b = a + gratio * (c-a)    # b will be inside a, c
    pot_min = b               # current minimum estimate

    #step before last, last step initizalized to b-a
    step_bl = step_l = np.abs(b-a)

    niter = 0

    #loops until tolerance is reached or max number of iterations is reached
    while ((np.abs(step_l) > tol) & (niter < maxiter)):

        #parabolic step
        pot_min = parabolic_step(func=func, a=a, b=b, c=c)  #new potential minimum

        #if the minimum is not in the interval, or step is greater than step before last step
        if ((np.abs(pot_min-b) < step_bl) & (a<=pot_min<=c)):

            if (pot_min<b):
                c = b
            else:
                a = b

            step_bl = step_l
            step_l = b - pot_min

            step = np.array([b, pot_min])
            plt.plot(step, func(step), color='black',label='parabolic step')
            plt.plot(step, func(step), '.', color='black')
            b = pot_min

        else:
            #golden section
            # choose point in the bigger interval
            if ((b - a) > (c - b)):
                pot_min = b
                b = b - gratio * (b - a)
            else:
                pot_min = b + gratio * (c - b)

            step_bl = step_l
            step_l = b - pot_min

            step = np.array([b, pot_min])
            plt.plot(step, func(step), color='red')
            plt.plot(step, func(step), '.', color='red')

            fb = func(b)
            fx = func(pot_min)

            if (fb < fx):
                c = pot_min
            else:
                a = b
                b = pot_min
        niter = niter + 1

    colors = ['b','r']
    texts = ['parabolic','golden section']
    patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts))]
    plt.legend(handles=patches, loc='upper right', ncol=1 )
    plt.title("Brent's Minimization")
    plt.xlabel("X")
    plt.ylabel("f(x)")
    plt.show()
    return b, func(b), niter

def func(x=None):
    return np.power((x-0.3),2) * np.exp(x)

def main():
    res = scipy.optimize.brent(func,brack=(0,1),full_output=True)
    smin, sval, siter, _ = res
    print("Scipy's Brent:")
    print("\tMinimum: ", smin)
    print("\tIterations", siter)

    bmin,bval,biter = brent(func,0,1)
    print("My Brent:")
    print("\tMinimum: ", bmin)
    print("\tIterations", biter)
    print("difference in minimum: ", np.abs(smin-bmin))


if __name__ == '__main__':
    main()

