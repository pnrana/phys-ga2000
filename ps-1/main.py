# Computational Physics
# Problem Set #1

import numpy as np
import matplotlib.pyplot as plt

def main():
    mu= 0  #mean
    sigma = 3   #standard deviation

    x_axis = np.arange(-10,10.01,0.01)

    y_axis = [0 for x in range(len(x_axis))]

    for i in range(len(x_axis)):
        y_axis[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x_axis[i] - mu)**2 / (2 * sigma**2))

    area = np.trapz(y_axis,x_axis)
    print("Normalized" if round(area,2)==1 else "Not Normalized")

    plt.plot(x_axis,y_axis,label='plot_Gaussian')
    plt.title('Normalized Gaussian Distribution (Mean = 0, Std_Dev = 3)')
    plt.xlabel('X')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
