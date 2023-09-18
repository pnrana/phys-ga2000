import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow,show
from matplotlib import colors

def main():
    
    Ndim=1000
    x = np.linspace(-2,2,Ndim)
    y = np.linspace(-2,2,Ndim)
    cx,cy = np.meshgrid(x,y)

    zx = np.float64(np.zeros((Ndim,Ndim)))
    zy = np.float64(np.zeros((Ndim,Ndim)))

    mask = np.zeros((Ndim,Ndim),dtype=int)
    mag_z = np.zeros((Ndim,Ndim))


    for i in range (0,100):
        nzx = np.float64(zx**2 - zy**2 +cx)
        nzy = np.float64(2*zx*zy+cy)
      
        mag_z[mask==i] = (nzx[mask==i]**2 + nzy[mask==i]**2)**(1/2)
        mask[mag_z<=2]+= 1 
        zx[mask>i] = nzx[mask>i]
        zy[mask>i] = nzy[mask>i]

    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    dplot = imshow(mask,origin="lower", cmap='jet',extent=extent)

    plt.title('Density plot of elements of the Mandelbrot set')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.colorbar()

    plt.savefig('mandel_iterated.pdf',dpi=2000)

if __name__ == "__main__":
    main()
