# Computational Physics
# Problem Set #3
# 3.4

import numpy as np
from random import random
import matplotlib.pyplot as plt


def main():
    # Constants
    NPb = 0  # Number of Lead atoms
    NTl = 1000  # Number of Thallium atoms
    tau = 3.053 * 60  # Half life of Thallium in seconds

    z = np.random.rand(NTl)
    decaytimes = -(tau/np.log2(2))*np.log2(1-z)
    decaytimes = np.sort(decaytimes)

    tmax = 1000  # Total time in seconds

    # Lists of plot points
    tpoints = np.arange(0.0, tmax)
    not_decay = []

    for t in tpoints:
        not_decay.append(np.count_nonzero(decaytimes > t))

    plt.figure()
    plt.plot(tpoints,not_decay,label='Atoms that have not decayed')
    plt.xlabel('Time (s)')
    plt.ylabel("Number of atoms")
    plt.title("Radioactive Decay of Thallium-209")
    plt.legend()
    plt.savefig('problem4.png')
    plt.show()




if __name__ == '__main__':
    main()

