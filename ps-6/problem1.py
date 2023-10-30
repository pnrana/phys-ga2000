#ps6

import numpy as np
import matplotlib.pyplot as plt
import astropy as ap
from astropy.io import fits
import time


def coefficients(N, R, eigvecs):
    # projecting residuals onto first 5 principal components to get coefficients
    return (np.dot(R, eigvecs[:, :N]))


def approx_spec(N, R, eigvecs, mean_spec, normalization):


    coeffs = coefficients(N, R, eigvecs)

    # multiplying the coefficients by the eigenspectra
    approx_spec = mean_spec[:, np.newaxis] + np.dot(coeffs, eigvecs[:, :N].T)

    # reapplying the original normalization constants
    approx_spec = np.multiply(approx_spec[:, :], normalization[:, np.newaxis])

    return approx_spec


def main():
    # a) Reading in the data
    hdu_list = ap.io.fits.open('specgrid.fits')
    logwave = hdu_list['LOGWAVE'].data  # logwave is in log10 Angstroms
    flux = hdu_list['FLUX'].data  # spectra of galaxies
    hdu_list.close()

    #---------------------------------(A)------------------------------------------#
    # hydrogen spectrum lines from Balmer transitions in angstroms
    balmer = [6563, 4861, 4340, 4102, 3970]
    balmer = np.log10(balmer)

    # plotting the spectra
    plt.figure(figsize=(10, 5))
    plt.title('Spectra of 5 galaxies')
    plt.xlabel('$log_{10} \lambda (\AA)$')
    plt.ylabel('flux')

    # plotting main curve
    for i in range(0, len(flux), int(len(flux) / 5)):
        plt.plot(logwave, flux[i])


    # plotting hydrogen spectra below the main curve
    for line in balmer:
        plt.axvline(x=line, color='r', linestyle='--')

    #adding legend
    plt.plot([], [], 'r--', label='Hydrogen spectrum')
    plt.plot([], [], 'k', label='Galaxy spectra')
    plt.legend()

    plt.savefig('ps6_1a.png')
    #plt.show()
    plt.clf()


    #---------------------------------(B)------------------------------------------#
    # normalizing fluxes so their integrals over wavelength are the same
    normalization_factor = np.sum(flux, axis=1)
    normalized_flux = flux[:,:] /normalization_factor[:,np.newaxis]


    #---------------------------------(C)------------------------------------------#
    #calculating residuals
    means = np.mean(normalized_flux, axis=1)
    residuals = normalized_flux[:,:] - means[:,np.newaxis]

    #---------------------------------(D)------------------------------------------#
    #PCA
    #recasting residuals as a matrix
    R = np.matrix(residuals)

    # timing the operation
    start = time.time()

    # calculating covariance matrix
    # note it needs to be R.transpose x R since we want the covariance matrix to be of dimension (N_wave x N_wave)
    # and R is of dimension (N_galaxies x N_wave)
    C = R.T @ R

    # calculating eigenvectors
    eigvals_cov, eigvecs_cov = np.linalg.eig(C)

    time_cov = time.time() - start

    # sorting eigenvectors
    idx = eigvals_cov.argsort()[::-1]
    eigvals_cov = eigvals_cov[idx]
    eigvecs_cov = eigvecs_cov[:, idx]

    # plotting the first five eigenvectors
    [plt.plot(logwave, eigvecs_cov[:, i], label=f'Eigenvector {i + 1}') for i in range(5)]

    plt.xlabel('$log_{10} \lambda (\AA)$')
    plt.ylabel('')
    plt.title("First five eigenvectors from covariance method")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('ps6_1d.png')
    plt.clf()

    #---------------------------------(E)------------------------------------------#
    #calculating eigenvalues with svd

    # timing the SVD method
    start = time.time()

    _, S, Vt = np.linalg.svd(residuals, full_matrices=False)

    eigvecs_svd = Vt.T
    eigvals_svd = S ** 2

    time_svd = time.time() - start



    # sorting eigenvectors
    idx = eigvals_svd.argsort()[::-1]
    eigvals_svd = eigvals_svd[idx]
    eigvecs_svd = eigvecs_svd[:, idx]

    # plotting the first five eigenvectors
    [plt.plot(logwave, eigvecs_svd[:, i], label=f'Eigenvector {i + 1}') for i in range(5)]

    plt.xlabel('$log_{10} \lambda (\AA)$')
    plt.title("First five eigenvectors from SVD")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('ps6_1e.png')
    plt.clf()

    #comparing the eigenvectors from the two methods
    for i in range(0, len(eigvecs_svd)):
        plt.plot(eigvecs_svd[:, i], eigvecs_cov[:, i], 'o')
    plt.xlabel('SVD eigenvalues')
    plt.ylabel('Covaraince eigenvalues')
    plt.title('Comparing eigenvectors from SVD and covariance method')
    #plt.show()
    plt.savefig('ps6_1e1.png')
    plt.clf()

    print("\nTime for covariance method:", time_cov)
    print("Time for SVD method:", time_svd)
    #---------------------------------(F)------------------------------------------#

    print("\nCondition number of R:", np.linalg.cond(R))
    print("Condition number of C:",np.linalg.cond(C))

    #---------------------------------(G)------------------------------------------#

    #approximating spectra with 5 principal components
    approx_5 = approx_spec(5, R, eigvecs_svd, means, normalization_factor)

    #plotting the approximated spectra
    plt.plot(logwave, approx_5[0, :].T, label='N=5')
    plt.plot(logwave, flux[0, :], label='Original')

    plt.xlabel('$log_{10} \lambda (\AA)$')
    plt.ylabel('Flux')
    plt.title("Approximated spectra with 5 principal components")
    plt.legend()
    #plt.show()
    plt.savefig('ps6_1g.png')
    plt.clf()

    #---------------------------------(H)------------------------------------------#
    #comparing c0, c1, c2
    coeffs = coefficients(5, R, eigvecs_svd)
    c0 = coeffs[:, 0]
    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]

    plt.plot(c0, c1, 'o', label='c0 vs c1')
    plt.plot(c0, c2, 'o', label='c0 vs c2')
    plt.xlabel('c0')
    plt.ylabel('c1, c2')
    plt.title('Comparing c0, c1, c2')
    plt.legend()
    #plt.show()
    plt.savefig('ps6_1h.png')
    plt.clf()

    #---------------------------------(I)------------------------------------------#
    squared_rs = []

    for N in range(1, 21):
        approx = approx_spec(N, R, eigvecs_cov, means, normalization_factor)

        rs = (flux - approx)

        squared_rs.append(np.mean(np.square(rs)))

    # Print the squared fractional residual for Nc = 20
    print('\nSquared fractional residual for N = 20:', squared_rs[-1])

    #plotting squared residuals
    plt.plot(range(1, 21), squared_rs)
    plt.xlabel('N')
    plt.ylabel('Squared residuals')
    plt.title('Squared residuals vs N')
    #plt.show()
    plt.savefig('ps6_1i.png')
    plt.clf()





if __name__ == '__main__':
    main()

