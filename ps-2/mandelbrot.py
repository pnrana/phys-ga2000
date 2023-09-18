import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow,show
from matplotlib import colors
import matplotlib.patches as mpatches

def main():
    #specify number of grid points in an axis
    Ndim=1000
    x = np.linspace(-2,2,Ndim)
    y = np.linspace(-2,2,Ndim)
    #make a mesh of all gridpoints that span the region
    cx,cy = np.meshgrid(x,y)

    #for Z values
    zx = np.float64(np.zeros((Ndim,Ndim)))
    zy = np.float64(np.zeros((Ndim,Ndim)))

    #mask to keep track of elements whose magnitude becomes >= 2
    mask = np.ones((Ndim,Ndim),dtype=bool)
    mag_z = np.zeros((Ndim,Ndim))


    for i in range (0,100):
        nzx = np.float64(zx**2 - zy**2 +cx)
        nzy = np.float64(2*zx*zy+cy)
      
        mag_z[mask] = (nzx[mask]**2 + nzy[mask]**2)**(1/2)
        mask[mag_z>=2] = 0
        zx[mask] = nzx[mask]
        zy[mask] = nzy[mask]

    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    dplot = imshow(mask,origin="lower", cmap='binary',extent=extent)

    plt.title('Density plot of elements of the Mandelbrot set')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    black_patch = mpatches.Patch(color='black', label='element of set')
    plt.legend(handles=[black_patch])

    plt.savefig('mandel.pdf',dpi=2000)


if __name__ == "__main__":
    main()
