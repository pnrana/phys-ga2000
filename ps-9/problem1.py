# Newman 9.8
# The schrodinger equation and the Crank-Nicolson method

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import animation
from banded import banded

def crank_nicolson(A,B,psi):
    psi_new = np.zeros(np.shape(psi),dtype=complex)

    ## Calculating v via matrix multiplication
    # v1 = B @ psi

    # Calculating v with the formula given in the problem
    v = np.zeros(np.shape(psi),dtype=complex)
    v[1:-1] = b1 * psi[1:-1] + b2 * (psi[2:] + psi[0:-2])

    # Checking both methods for calculating v
    # if (v==v1):
    #     print("They match")
    # else:
    #     print("Not match")

    # solving for x
    psi_new = np.linalg.solve(A, v)
    psi_new[0] = 0
    psi_new[-1] = 0
    return psi_new

def animate(x,psi,save_gif=False):
    # animation things I don't fully understand
    # copied from lecture jupyter notebook
    fig, ax = plt.subplots()

    ax.set_title("Particle in a box")
    ax.set_xlim((0, L))
    ax.set_ylim((- 1, 1))
    ax.set_ylabel("$\psi$")
    ax.set_xlabel('x')

    line, = ax.plot([], [], lw=2)

    # animation helper
    def frame(i):
        line.set_data(x, np.real(psi[i, :])**2)
        return (line,)

    def init():
        line.set_data([], [])
        return (line,)

    anim = animation.FuncAnimation(fig, frame, init_func=init,
                                   frames=nsteps, interval=4,
                                   blit=True)

    if save_gif:
    # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=32,metadata=dict(artist='Me'),bitrate=1500)
        anim.save('pbox.gif', writer=writer)
    else:
        plt.show()

def main():
    #gridpoints
    x = np.linspace(0, L, N)

    #wavefunction array where each row is a wavefunction at a timestep
    psi = np.zeros((nsteps, N), dtype=complex)

    # setting wavefunction at t=0
    pstart = np.exp(-(x - x0) ** 2 / (2 * (sig ** 2))) * np.exp(1j * k * x)

    #print((x-x0)**2/ (2 * (sig ** 2)))
    psi[0, :] = pstart

    # boundary conditions
    psi[:, 0] = psib
    psi[:, -1] = psib


    # making the matrices
    A = np.zeros((N, N), dtype=complex)
    B = np.zeros((N, N), dtype=complex)

    #settings values for the main diagonal(except last element), lower and upper diagonals
    for i in range (N-1):
        #main diagonal
        A[i, i] = a1
        B[i, i] = b1
        #upper diagonal
        A[i, i+1] = a2
        B[i, i+1] = b2
        #lower diagonal
        A[i+1, i] = a2
        B[i+1, i] = b2

    #setting values for the last main diagonal element
    A[N-1, N-1] = a1
    B[N-1, N-1] = b1

    for i in range(nsteps-1):
        psi[i+1,:] = crank_nicolson(A,B,psi[i])

    animate(x,psi)


if __name__ == '__main__':
    #user defined variables
    nsteps = 1000    # number of timesteps
    h = 1e-18        # timestep in seconds

    # constants
    M = 9.109e-31   # mass of electron in kg
    L = 1e-8        # length of the box in meters
    N = 1000        # number of spatial slices
    a = L / N       # Grid spacing

    # Boundary conditions
    psib = 0 + 0j   # wavefunction at boundaries

    # constants for wavefunction calculation
    x0 = np.float64(L / 2)      # x_0
    sig = 1e-10                 # sigma in m
    k = 5e10                    # k in m^-1

    # constants for the  matrices
    hbar = sp.constants.hbar
    a1 = 1 + ((h * 1j * hbar) / (2 * M * a ** 2))
    a2 = (-1) * h * 1j * hbar / (4 * M * a ** 2)
    b1 = 1 - ((h * 1j * hbar) / (2 * M * a ** 2))
    b2 = (1) * h * 1j * hbar / (4 * M * a ** 2)

    main()