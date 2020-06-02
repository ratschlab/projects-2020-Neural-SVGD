import jax.numpy as np
from jax import vmap
import warnings

import utils

"""A collection of positive definite kernel functions written using Jax."""

def _ard(x, y, logh):
    """
    IN:
    * x, y: np arrays of shape (d,)
    * logh: np array of shape (d,), or scalar. represents log of bandwidth parameter (so can be negative or zero).

    OUT:
    Scalar value of the ARD kernel evaluated at (x, y, h).
    """
    x, y = np.array(x), np.array(y)
    if x.shape != y.shape:
        raise ValueError(f"Shapes of particles x and y need to match. Recieved shapes x: {x.shape}, y: {y.shape}")
    if x.ndim > 1:
        raise ValueError(f"Input particles x and y can't have more than one dimension. Instead they have rank {x.ndim}")

    logh = np.array(logh)
    if logh.ndim > 1:
        raise ValueError(f"Bandwidth needs to be a scalar or a d-dim vector. Instead it has shape {logh.shape}")
    elif logh.ndim == 1:
        assert x.shape == logh.shape

    h = np.exp(logh)
    return np.exp(- np.sum((x - y)**2 / h) / 2)

def ard(h):
    return lambda x, y: _ard(x, y, h)

def _ard_m(x, y, sigma):
    """
    Arguments:
    * x, y : array-like. Shape (d,)
    * sigma: array-like. Shape (d, d). Must be positive definite.

    Returns:
    Scalar given by
    \[ e^{- 1/2 (x - y)^T \Sigma^{-1} (x - y)} \]
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"Shapes of particles x and y need to match. Recieved shapes x: {x.shape}, y: {y.shape}")
    elif x.ndim > 1 or x.ndim == 0:
        raise ValueError(f"Input particles x and y need to have shape (d,). Instead received shape {x.shape}")
    sigma = np.asarray(sigma)
    d = x.shape[0]
    if sigma.ndim != 2 and d != 1:
        raise ValueError(f"Sigma needs to be a square matrix. Instead, received shape {sigma.shape}.")

    inv = np.linalg.inv(sigma)
    return np.exp(- np.matmul(np.matmul(x - y, inv), x - y) / 2)

def ard_m(sigma):
    return lambda x, y: _ard_m(x, y, sigma)

def median_heuristic(x):
    """
    Heuristic for choosing ARD bandwidth.

    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: bandwidth parameter for RBF kernel, based on the heuristic from the SVGD paper.
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

