# #ps7
# #logsitic regression

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import scipy.optimize as optimize
import numpy as np

#hessian function using jax
def hessian(f):
   return jax.jacfwd(jax.grad(f))

# Logistic function
def logistic_function(x, b0, b1):
    return 1 / (1 + jnp.exp(-(b0 + b1 * x)))

#negative log likelihood function
def negloglike(beta,x,y):
    b0 = beta[0]
    b1 = beta[1]
    z = b0 + (b1 * x)
    lnpi = jnp.log(1 + jnp.exp(z)) - (y * z)
    nll = jnp.sum(lnpi)
    return nll

def main():
    #importing csv file
    cols = np.genfromtxt('survey.csv', delimiter=',', dtype=str, max_rows=1)
    data = np.genfromtxt('survey.csv', delimiter=',', skip_header=1)

    x = jnp.array(data[:,0])
    y = jnp.array(data[:,1])

    #initial guesses for beta0 and beta1
    initial_betas = jnp.array([0.,0.])

    # Gradient of the negative log-likelihood
    nll_grad = jax.grad(negloglike)

    r = optimize.minimize(fun = negloglike, x0 = initial_betas, jac=nll_grad, args= (x,y), method='BFGS',tol=1e-6)

    print("Maximum likelihood values:")
    print(r.x)
    print("Number of function evaluations: {n}".format(n=r.nfev))
    print("Number of iterations: {n}".format(n=r.nit))

    ages = np.linspace(0, 100, 100)

    plt.plot(data[:, 0], data[:, 1],'o')
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    probab = logistic_function(ages, r.x[0], r.x[1])

    plt.plot(ages, probab, color='red', label='Logistic model')
    plt.legend(loc='center right')
    plt.title('Do people "Be Kind, Rewind"?')
    plt.show()

    #finding the hessian matrix
    h = hessian(negloglike)
    hmat = h(r.x,x,y)

    #covariance matrix
    cov = jnp.linalg.inv(hmat)
    print("Covariance matrix:")
    print(cov)

    #finding the standard errors
    std_err = jnp.sqrt(jnp.diag(cov))
    print("Standard errors:")
    print(std_err)

    #finding the correlation matrix
    corr = cov / jnp.outer(std_err, std_err)
    print("Correlation matrix:")
    print(corr)




if __name__ == '__main__':
    main()

