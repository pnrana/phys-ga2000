# Computational Physics
# Problem Set #3
# 3.3

import numpy as np
from random import random
import matplotlib.pyplot as plt



def main():
    # Constants
    NBiI = 10000   # Number of Bismuth 213 isotopes
    NPb = 0  # Number of Lead atoms
    NTl = 0  #Number of Thallium atoms
    NBiS = 0 # Number of stable Bi 209

    tauBiI = 46 * 60    # Half life of Bismuth 213 in seconds
    tauTl = 2.2 * 60    # Half life of Thallium in seconds
    tauPb = 3.3 * 60    # Half life of Lead in seconds
    h = 1.0             # Size of time-step in seconds
    pBiTl = 0.0209      # Probability of Bismuth 213 decaying into Thallium
    pBiI = 1 - 2 ** (-h / tauBiI)   # Probability of decay in one step
    pTl = 1 - 2 ** (-h / tauTl)     # Probability of decay in one step
    pPb = 1 - 2 ** (-h / tauPb)     # Probability of decay in one step

    tmax = 20000        # Total time in seconds

    # Lists of plot points
    tpoints = np.arange(0.0,tmax,h)
    BiIpoints = []
    Tlpoints = []
    Pbpoints = []
    BiSpoints = []

    for t in tpoints:
        BiIpoints.append(NBiI)
        Tlpoints.append(NTl)
        Pbpoints.append(NPb)
        BiSpoints.append(NBiS)

        # Calculate the number of PB atoms that decay
        decay = 0
        for i in range(NPb):
            if random() < pPb:
                decay += 1
        NPb -= decay
        NBiS += decay

        # Calculate the number of Tl atoms that decay
        decay = 0
        for i in range(NTl):
            if random() < pTl:
                decay += 1
        NTl -= decay
        NPb += decay

        # Calculate the number of Bi atoms that decay
        decay = 0
        for i in range(NBiI):
            if random() < pBiI:
                decay += 1
                if random() < pBiTl:
                    NTl += 1
                else:
                    NPb += 1
        NBiI -= decay

    # Make the graph
    plt.figure()
    plt.plot(tpoints,BiIpoints,label='Bi_213')
    plt.plot(tpoints,Tlpoints,label='Tl_209')
    plt.plot(tpoints,Pbpoints,label='Pb_209')
    plt.plot(tpoints, BiSpoints,label='Bi_209')
    plt.legend()
    plt.xlabel ("Time (s)")
    plt.ylabel("Number of atoms")
    plt.title("Radioactive Decay Chain")
    plt.savefig('problem3.png')
    plt.show()


if __name__ == '__main__':
    main()

