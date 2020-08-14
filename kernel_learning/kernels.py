import jax.numpy as np
from jax import vmap
import utils

"""A collection of positive definite kernel functions.
Every kernel takes as input two jax scalars or arrays x, y of shape (d,),
where d is the particle dimension, and outputs a scalar.
"""
def _check_xy(x, y):
    x, y = [np.asarray(v) for v in (x, y)]
    if x.shape != y.shape:
        raise ValueError(f"Shapes of particles x and y need to match. "
                         f"Recieved shapes x: {x.shape}, y: {y.shape}")
    elif x.ndim > 1:
        raise ValueError(f"Input particles x and y can't have more than one "
                         f"dimension. Instead they have rank {x.ndim}")
    return x, y

def _check_bandwidth(bandwidth):
    bandwidth = np.squeeze(np.asarray(bandwidth))
    if bandwidth.ndim > 1:
        raise ValueError(f"Bandwidth needs to be a scalar or a d-dim vector. "
                         f"Instead it has shape {bandwidth.shape}")
    elif bandwidth.ndim == 1:
        assert x.shape == bandwidth.shape
    return bandwidth

def get_rbf_kernel(bandwidth):
    bandwidth = _check_bandwidth(bandwidth)
    def rbf(x, y):
        x, y = _check_xy(x, y)
        return np.exp(- np.sum((x - y)**2 / bandwidth**2) / 2)
    return rbf

def get_tophat_kernel(bandwidth):
    bandwidth = _check_bandwidth(bandwidth)
    def tophat(x, y):
        x, y = _check_xy(x, y)
        return np.squeeze(np.where(np.linalg.norm(x-y)<bandwidth, 1., 0.))
    return tophat

def get_rbf_kernel_logscaled(logh):
    logh = np.asarray(logh)
    bandwidth = np.exp(logh/2) # TODO remove 1/2
    return get_rbf_kernel(bandwidth)

def get_tophat_kernel_logscaled(logh):
    logh = np.asarray(logh)
    bandwidth = np.exp(logh/2) # TODO remove 1/2
    return get_tophat_kernel(bandwidth)

def constant_kernel(x, y):
    _check_xy(x, y)
    return np.array(1.)

def funnelize(v):
    """If v is standard 2D normal, then
    funnelize(v) is distributed as Neal's Funnel."""
    *x, y = v
    x, y = np.asarray(x), np.asarray(y)
    return np.append(x*np.exp(3*y/2), 3*y)

def defunnelize(z):
    """Inverse of funnelize."""
    *x, y = z
    x, y = np.asarray(x), np.asarray(y)
    return np.append(x*np.exp(-y/2), y/3)

def get_funnel_kernel(bandwidth):
    rbf = get_rbf_kernel(bandwidth)
    def funnel_kernel(x, y):
        return rbf(defunnelize(x), defunnelize(y))
    return funnel_kernel

### Utils
def median_heuristic(x):
    """
    Heuristic for choosing ARD bandwidth.

    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: bandwidth parameter for RBF kernel, based on the
    heuristic from the SVGD paper.
    Note: assumes k(x, y) = exp(- (x - y)^2 / h / 2)
    """
    if x.ndim == 2:
        return vmap(median_heuristic, 1)(x)
    elif x.ndim == 1:
        n = x.shape[0]
        medsq = np.median(utils.squared_distance_matrix(x))
        h = medsq / np.log(n) / 2
        return h
    else:
        raise ValueError("Shape of x has to be either (n,) or (n, d)")


def _ard_m(x, y, sigma):
    """
    Arguments:
    * x, y : array-like. Shape (d,)
    * sigma: array-like. Shape (d, d). Must be positive definite.

    Returns:
    Scalar given by
    \[ e^{- 1/2 (x - y)^T \Sigma^{-1} (x - y)} \]
    """
    x, y = _check_xy(x, y)
    sigma = np.asarray(sigma)
    d = x.shape[0]
    if sigma.ndim != 2 and d != 1:
        raise ValueError(f"Sigma needs to be a square matrix. Instead, received shape {sigma.shape}.")

    inv = np.linalg.inv(sigma) # TODO better: cholesky. also check PD
    return np.exp(- np.matmul(np.matmul(x - y, inv), x - y) / 2)

def ard_m(sigma):
    return lambda x, y: _ard_m(x, y, sigma)
