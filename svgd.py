import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

from utils import ard, squared_distance_matrix

def phi_j(x, y, logp, kernel):
    """
    IN:
    * x, y: np arrays of shape (d,)
    * kernel: callable, computes the kernel k(x, y)
    * logp: callable, computes log of a differentiable pdf p(x) given input value x

    OUT:
    * np array of shape (d,):
    \nabla_x log(p(x)) * k(x, y) + \nabla_x k(x, y).
    This means that phi(x_i) = \sum_j phi_j(x_j, x_i)
    """
    assert x.ndim == 1 and y.ndim == 1
    return grad(logp)(x) * kernel(x, y) + grad(kernel)(x, y)
phi_j_batched = vmap(phi_j, (0, 0, None, None), 0)

def update(x, logp, stepsize, kernel_params):
    """
    IN:
    * x: np array of shape n x d
    * logp: callable, log of a differentiable pdf p
    * stepsize: scalar > 0
    * kernel_params: np array (or dict?) fed to kernel(x, y, kernel_params)

    OUT:
    xnew = x + stepsize * \phi^*(x)
    that is, xnew is an array of shape n x d. The entries of x are the updated particles.

    note that this is an inefficient way to do things, since we're computing k(x, y) twice for each x, y combination.
    """
    assert x.ndim == 2
    kernel = lambda x, y: ard(x, y, kernel_params)

    xnew = []
    n = x.shape[0]
    for i, xi in enumerate(x):
        repeated = np.tile(xi, (n, 1))
        xnew.append(stepsize * np.sum(phi_j_batched(x, repeated, logp, kernel), axis = 0))
    xnew = np.array(xnew)
    xnew += x

    return xnew

update = jit(update, static_argnums=(1,)) # logp is static. When logp changes, jit recompiles.


def svgd(x, logp, stepsize, L, kernel_param, kernel_param_update_rule=None):
    """
    IN:
    * x is an np array of shape n x d
    * logp is the log of a differentiable pdf p (callable)
    * stepsize is a float
    * kernel_param is a positive scalar: bandwidth parameter for RBF kernel
    * L is an integer (number of iterations)
    * kernel_param_update_rule is a callable that takes the updated particles as input and outputs an updated set of kernel parameters. If supplied, the argument kernel_param will be ignored.

    OUT:
    * Updated particles x (np array of shape n x d) after L steps of SVGD
    * dictionary with logs
    """
    assert x.ndim == 2
    if kernel_param_update_rule is not None and kernel_param is not None:
        raise ValueError("When kernel_param_update_rule is supplied, kernel_param should be None (as it's not used in that case).")
    elif kernel_param_update_rule is None and kernel_param is None:
        raise ValueError("When kernel_param_updater_rule is None, you need to supply a value for the kernel parameter.")
    else:
        pass

    log = {
        "kernel_params": [],
        "particle_mean": [np.mean(x, axis=0)],
        "particle_var": [np.var(x, axis=0)]
    }

    for i in range(L):
        if kernel_param_update_rule is not None:
            kernel_param = kernel_param_update_rule(x)
            log["kernel_params"].append(kernel_param)
        else:
            pass

        x = update(x, logp, stepsize, kernel_param)
        log["particle_mean"].append(np.mean(x, axis=0))
        log["particle_var"].append(np.var(x, axis=0))

        if np.any(np.isnan(x)):
            log["errors"] = f"NaN produced at iteration {i}"
            break
    return x, log

@jit
def kernel_param_update_rule(x):
    """
    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: Updated bandwidth parameter for RBF kernel, based on update rule from the SVGD paper.
    """
    if x.ndim == 2:
        return vmap(kernel_param_update_rule, 1)(x)
    elif x.ndim == 1:
        n = x.shape[0]
        h = np.median(squared_distance_matrix(x)) / np.log(n)
        return h
    else:
        raise ValueError("Shape of x has to be either (n,) or (n, d)")
