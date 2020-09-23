import jax.numpy as np
from jax import vmap, grad
from jax.scipy import stats
import utils

"""A collection of positive definite kernel functions.
Every kernel takes as input two jax scalars or arrays x, y of shape (d,),
where d is the particle dimension, and outputs a scalar.
"""
def _check_xy(x, y, dim=None):
    x, y = [np.asarray(v) for v in (x, y)]
    if x.shape != y.shape:
        raise ValueError(f"Shapes of particles x and y need to match. "
                         f"Recieved shapes x: {x.shape}, y: {y.shape}")
    elif x.ndim > 1:
        raise ValueError(f"Input particles x and y can't have more than one "
                         f"dimension. Instead they have rank {x.ndim}")
    if dim is not None:
        if dim > 1:
            if not x.shape[0] == dim:
                raise ValueError(f"x must have shape {(dim,)}. Instead received "
                                 f"shape {x.shape}.")
        elif dim == 1:
            if not (x.ndim==0 or x.shape[0]==dim):
                raise ValueError(f"x must have shape (1,) or scalar. Instead "
                                 f"received shape {x.shape}.")
        else: raise ValueError(f"dim must be a natural nr")
    return x, y

def _check_bandwidth(bandwidth, dim=None):
    bandwidth = np.squeeze(np.asarray(bandwidth))
    if bandwidth.ndim > 1:
        raise ValueError(f"Bandwidth needs to be a scalar or a d-dim vector. "
                         f"Instead it has shape {bandwidth.shape}")
    elif bandwidth.ndim == 1:
        pass

    if dim is not None:
        if not (bandwidth.ndim == 0 or bandwidth.shape[0] in (0, 1, dim)):
            raise ValueError(f"Bandwidth has shape {bandwidth.shape}.")
    return bandwidth

def _normalizing_factor(bandwidth):
    if bandwidth.ndim==1 or bandwidth.shape[0]==1:
        return np.sqrt(2*np.pi)
    else:
        d = bandwidth.shape[0]
        return (2*np.pi)**(-d/2) * 1/np.sqrt(np.prod(bandwidth))


def get_rbf_kernel(bandwidth, normalize=False, dim=None):
    bandwidth = _check_bandwidth(bandwidth, dim)
    def rbf(x, y):
        x, y = _check_xy(x, y, dim)
        if normalize:
            return np.prod(stats.norm.pdf(x, loc=y, scale=bandwidth))
        else:
            return np.exp(- np.sum((x - y)**2 / bandwidth**2) / 2)
    return rbf

def get_tophat_kernel(bandwidth, normalize=False, dim=None):
    bandwidth = _check_bandwidth(bandwidth, dim)
    volume = np.prod(2*bandwidth)
    def tophat(x, y):
        x, y = _check_xy(x, y, dim)
        if normalize:
            return np.squeeze(np.where(np.all(np.abs(x - y) < bandwidth), 1/volume, 0.))
        else:
            return np.squeeze(np.where(np.all(np.abs(x - y) < bandwidth), 1., 0.))
    return tophat

def get_rbf_kernel_logscaled(logh, normalize=False):
    logh = np.asarray(logh)
    bandwidth = np.exp(logh/2) # TODO remove 1/2
    return get_rbf_kernel(bandwidth, normalize)

def get_tophat_kernel_logscaled(logh):
    logh = np.asarray(logh)
    bandwidth = np.exp(logh/2) # TODO remove 1/2
    return get_tophat_kernel(bandwidth)

def constant_kernel(x, y):
    """Returns 1."""
    _check_xy(x, y)
    return np.array(1.)

def char_kernel(x, y):
    """Returns 1 if x==y, else 0"""
    _check_xy(x, y)
    return np.squeeze(np.where(x==y, 1., 0.))

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

def scalar_product_kernel(x, y):
    """k(x, y) = x^T y"""
    return np.inner(x, y)

def get_imq_kernel(alpha: float=1, beta: float=-0.5):
    """
    alpha > 0
    beta \in (-1, 0)
    Returns:
    kernel k(x, y) = (alpha + ||x - y||^2)^beta
    """
    def inverse_multi_quadratic_kernel(x, y):
        return (alpha + utils.normsq(x - y))**beta
    return inverse_multi_quadratic_kernel

def get_inverse_log_kernel(alpha: float):
    def il_kernel(x, y):
        return (alpha + np.log(1 + utils.normsq(x - y)))**(-1)
    return il_kernel

def get_imq_score_kernel(alpha: float, beta: float, logp: callable):
    """
    Arguments:
    alpha > 0
    beta \in (-1, 0)
    logp computes log p(x)

    Returns:
    kernel k(x, y) = (alpha + ||\nabla \log p(x) - \nabla \log p(y)||^2)^beta
    """
    def imq_score_kernel(x, y):
        return (alpha + utils.normsq(grad(logp)(x) - grad(logp)(y)))**beta
    return imq_score_kernel

### Utils
def median_heuristic(x):
    """
    Heuristic for choosing RBF bandwidth.

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