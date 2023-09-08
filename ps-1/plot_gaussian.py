# Computational Physics
# Problem Set #1

import numpy as np
import matplotlib.pyplot as plt


def main():
    mu= 0  #mean
    sigma = 3   #standard deviation
    x_min = -10 
    x_max = 10
    
    #creating a numpy array initialized with 2000 x values over the range [-10,10]
    x_axis = np.arange(x_min,x_max,abs(x_min-x_max)/2000)
    
        
    #calculating the PDF of the gaussian distribution
    y_axis = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x_axis - mu)**2 / (2 * sigma**2))

    #normalizing the distribution by dividing the PDF by integral of the PDF over the range
    normalized = y_axis/ np.trapz(y_axis,x_axis)
    #print(np.trapz(y_axis,x_axis))

    plt.plot(x_axis,normalized)
    plt.title('Gaussian Distribution')
    plt.xlabel('X')
    plt.ylabel('Probability Density')
    plt.text(5.1,0.11,r'$\mu=0,\ \sigma=3$', fontsize = 15)
    plt.grid()

    #plt.show()
    plt.savefig("gaussian.png")
    
    

if __name__ == '__main__':
    main()

