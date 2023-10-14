#ps5
#problem 2
#Fitting Data

import numpy as np
import matplotlib.pyplot as plt

#SVD function
def svd(y,x,n):
    '''
    This function takes in a signal and time array and returns a model of the signal
    :param y: np.array
        numpy array of the dependant values
    :param x: np.array
        numpy array of the independent values
    :param n: int
        order of the polynomial to fit
    :return signal_m: np.array
        numpy array of the fitted curve
    '''
    A = np.zeros((len(x), (n+1)))

    for i in range(0, n+1):
         A[:, i] = x ** i

    (u, w, vt) = np.linalg.svd(A, full_matrices=False)

    ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

    coeffs = ainv.dot(y)
    return(A.dot(coeffs))

def svd_harmonics(signal,time,n):
    '''
    This function takes in a signal and time array and returns a model of the signal using sin and cos functions
    '''

    period = (np.max(time) - np.min(time)) / 2

    # Design matrix with sin and cos functions (2n columns) plus a zero offset point (1 column)
    A = np.zeros((len(time), 2 * n + 1))
    A[:, 0] = 1

    # iterate over the columns starting from 1->2n+1 inserting sin and cos functions in alternating columns
    for i in range(1, n + 1):
        A[:, 2 * i - 1] = np.sin(2 * np.pi * i * time / period)
        A[:, 2 * i] = np.cos(2 * np.pi * i * time / period)

    (u, w, vt) = np.linalg.svd(A, full_matrices=False)

    ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

    coeffs = ainv.dot(signal)
    return (A.dot(coeffs))

def main():
    #read from signal.dat file and skip header
    data = np.genfromtxt("signal.dat",delimiter='|',skip_header=1,usecols=(1,2))

    # Sort the numpy array based on the time column
    data = data[data[:, 0].argsort()]

    #splitting up the columns for better readability
    time = data[:,0]
    signal = data[:,1]

    #a) plotting the data
    plt.plot(time,signal,'.')
    plt.title("Signal vs Time")
    plt.ylabel("Signal")
    plt.xlabel("Time")
    #plt.show()
    #plt.savefig("ps5p2a.png",dpi=300)
    plt.clf()

    #b) Scaling the time values
    print("Range in time values:")
    print("Before rescaling: ", np.format_float_scientific(np.max(time)-np.min(time)))
    time = (time-np.mean(time))/np.std(time)
    print("After resclaing: ",np.format_float_scientific(np.max(time)-np.min(time)),'\n')

    #SVD for a third order polynomial fit
    signal_m=svd(signal,time,3)
    plt.title("3rd order polynomial fit")
    plt.plot(time,signal_m,label="model")
    plt.plot(time,signal,'.',label= "data")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    #plt.savefig("ps5p2b.png",dpi=300)
    #plt.show()
    plt.clf()

    #c) plotting the residuals
    residuals = signal-signal_m
    print("3rd order polynomial fit residuals:")
    print("Mean Residual: ",np.mean(residuals))
    print("Standard Deviation of Residuals: ",np.std(residuals),'\n')
    #std dev of residuals is higher than uncertainty in the data->model's errors are greater than what we expect from uncertainty in the data
    plt.plot(time,residuals,'.',label="data-model")
    plt.title("Residuals")
    plt.ylabel("$\Delta$ Signal")
    plt.xlabel("Time")
    plt.legend()
    #plt.savefig("ps5p2c.png",dpi=300)
    #plt.show()
    plt.clf()

    #d) trying a much higher order polynomial

    #lets check the condition numbers for different orders
    cond_nums = []
    min_n = 3
    max_n = 37
    interval = 1

    plt.subplot(2, 1, 1)

    for n in range(min_n,max_n+1,interval):
        A = np.zeros((len(time), (n + 1)))

        for i in range(0, n + 1):
            A[:, i] = time ** i

        cond = np.linalg.cond(A)
        cond_nums.append(cond)
        signal_m=svd(signal,time,n)

        if n%5==0:
            plt.plot(time,signal_m+n,label="n=%d" %n)

    plt.title("Polynomial Fits")
    plt.ylabel("Signal")
    plt.xlabel("Time")
    plt.legend(shadow=True, fancybox=True,fontsize='small',reverse=True,loc='center right')


    plt.subplot(2,1,2)
    plt.plot(range(min_n,max_n+1,interval),cond_nums,label="Condition Number")
    plt.plot(range(min_n, max_n + 1,interval), (np.ones(len(range(min_n, max_n + 1,interval))) * np.finfo(signal.dtype).eps) ** (-1),
             label="Machine Precision")
    plt.xlabel("Order of Polynomial")
    plt.ylabel("Condition Number")
    #putting x range ticks for every 2nd value
    plt.xticks(range(min_n, max_n + 1,2))
    plt.title("Condition Number vs Order of Polynomial")
    plt.legend()
    plt.tight_layout(h_pad=2)
    #plt.savefig("ps5p2d.png",dpi=300)
    #plt.show()
    plt.clf()


    #n=35 is the highest order polynomial that can be fit without the condition number being too high
    signal_m = svd(signal, time, 27)
    plt.plot(time, signal_m, label="model")
    plt.plot(time, signal, '.', label="data")
    plt.title("33rd order polynomial fit")
    plt.ylabel("Signal")
    plt.xlabel("Time")
    plt.tight_layout(h_pad=0)
    plt.legend()
    #plt.show()
    #plt.savefig("ps5p2d_fit.png", dpi=300)
    plt.clf()

    #checking the residuals
    residuals = signal - signal_m
    print("27th order polynomial fit residuals:")
    print("Mean Residual: ", np.mean(residuals))
    print("Standard Deviation of Residuals: ", np.std(residuals),'\n')


    #e) trying to fit a set of sin and cos functions plus a zero offset point

    # Determine the fundamental period and frequency
    period = (np.max(time) - np.min(time))/2


    #checking the condition numbers for different orders
    num=500
    interval=5

    cond_nums = []

    for n in range(400,num+1,interval):
        A = np.zeros((len(time), 2 * n + 1))
        A[:, 0] = 1

        for i in range(1, n + 1):
            A[:, 2 * i - 1] = np.sin(2 * np.pi * i * time / period)
            A[:, 2 * i] = np.cos(2 * np.pi * i * time / period)

        cond = np.linalg.cond(A)
        cond_nums.append(cond)

    plt.plot(range(400,num+1,interval),cond_nums,label="Condition Number")
    plt.plot(range(400, num + 1), (np.ones(len(range(400, num + 1))) * np.finfo(float).eps) ** (-1),
             label="Machine Precision")
    plt.xlabel("Number of Harmonics")
    plt.ylabel("Condition Number")
    plt.title("Condition Number vs Number of Harmonics")
    plt.legend()
    #plt.savefig("ps5p2e.png",dpi=300)
    #plt.show()
    plt.clf()

    #n=500 is the highest order polynomial that can be fit without the condition number being too high

    n = 300

    signal_m=svd_harmonics(signal,time,n)

    plt.plot(time,signal_m,label="model")
    plt.plot(time,signal,'.',label= "data")
    plt.title("sin and cos fit")
    plt.ylabel("Signal")
    plt.xlabel("Time")
    plt.legend()
    #plt.savefig("ps5p2e_fit.png",dpi=300)
    #plt.show()
    plt.clf()

    #checking the residuals
    residuals = signal - signal_m
    print("sin and cos fit residuals for %s harmonics:" %n)
    print("Mean Residual: ", np.mean(residuals))
    print("Standard Deviation of Residuals: ", np.std(residuals))
    #residuals are much smaller than the uncertainty in the data

    #smoothing the data in signal_m
    signal_m = np.convolve(signal_m, np.ones(10) / 10, mode='same')

    plt.plot(time, signal, '.', label="data")
    plt.plot(time, signal_m, label="model")
    plt.title("sin and cos fit")
    plt.ylabel("Signal")
    plt.xlabel("Time")
    plt.legend()
    #plt.show()
    #plt.savefig("ps5p2e_fit_smooth.png", dpi=300)
    plt.clf()



if __name__ == '__main__':
    main()

