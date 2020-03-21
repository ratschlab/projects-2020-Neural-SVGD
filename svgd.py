import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import matplotlib.pyplot as plt

from utils import normsq, single_rbf, pairwise_distances

def phi_j(x, y, logp, kernel):
    """
    IN:
    x and y are arrays of length d
    kernel is a callable that computes the kernel k(x, y, kernel_params)
    logp is the log of a differentiable pdf p

    OUT:
    \nabla_x log(p(x)) * k(x, y) + \nabla_x k(x, y)
    that is, phi(x_i) = \sum_j phi_j(x_j, x_i)
    """
    assert x.ndim == 1 and y.ndim == 1
    return grad(logp)(x) * kernel(x, y) + grad(kernel)(x, y)

phi_j_batched = vmap(phi_j, (0, 0, None, None), 0)

def update(x, logp, stepsize, kernel_params):
    """
    IN:
    x is an np array of shape n x d
    logp is the log of a differentiable pdf p
    stepsize is a float
    kernel_params are a set of parameters for the kernel

    OUT:
    xnew = x + stepsize * \phi^*(x)
    that is, xnew is an array of shape n x d. The entries of x are the updated particles.

    note that this is an inefficient way to do things, since we're computing k(x, y) twice for each x, y combination.
    """
    assert x.ndim == 2
    kernel = lambda x, y: single_rbf(x, y, kernel_params)
#     kerneltest = lambda x, y: np.exp(- normsq(x - y) / (2 * kernel_params ** 2))
#     assert kerneltest(x[0], x[1]) == kernel(x[0], x[1])

    xnew = []
    n = x.shape[0]
    for i, xi in enumerate(x):
        repeated = np.tile(xi, (n, 1))
        xnew.append(stepsize * np.sum(phi_j_batched(x, repeated, logp, kernel), axis = 0))
    xnew = np.array(xnew)
    xnew += x

    return xnew

update = jit(update, static_argnums=(1,)) # logp is static


def svgd(x, logp, stepsize, kernel_params, L, update_kernel_params=False, kernel_param_update_rule=None):
    """
    x is an np array of shape n x d
    logp is the log of a differentiable pdf p (callable)
    stepsize is a float
    kernel is a differentiable function k(x, y, h) that computes the rbf kernel (callable)
    L is an integer (number of iterations)

    if update_kernel_params is True, then kernel_param_update_rule must be given.
    kernel_param_update_rule is a callable that takes xnew as input and outputs an updated set of kernel parameters.

    OUT:
    Updated particles x (np array of shape n x d) after L steps of SVGD
    """
    assert x.ndim == 2
    log = {
        "kernel_params": [kernel_params],
        "particle_mean": [np.mean(x)],
        "particle_var": [np.var(x)]
    }

    for i in range(L):
        x = update(x, logp, stepsize, kernel_params)
        log["particle_mean"].append(np.mean(x))
        log["particle_var"].append(np.var(x))

        if np.any(np.isnan(x)):
            raise ValueError(f"NaN produced at iteration {i}")
        if update_kernel_params:
            kernel_params = kernel_param_update_rule(x)
            log["kernel_params"].append(kernel_params)
    return x, log
