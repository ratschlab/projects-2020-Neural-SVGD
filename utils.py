import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.ops import index_update, index

############ Kernel
@jit
def normsq(x):
    assert x.ndim == 1
    x = np.array(x)
    return np.vdot(x, x)

v_normsq = vmap(normsq) # outputs vector of norms
vv_normsq = vmap(v_normsq)

def single_rbf(x, y, h):
    """
    x and y are d-dimensional arrays
    h is a scalar parameter, h > 0
    """
    assert x.ndim == 1
    assert y.ndim == 1
    return np.exp(- normsq(x - y) / (2 * h))

@jit
def ard(x, y, h):
    """
    IN:
    x, y, h: np arrays of shape (d,)

    OUT:
    scalar kernel(x, y, h)
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert h.ndim == 1 or h.ndim == 0
    assert x.shape == y.shape
    if h.ndim == 1:
        assert x.shape == h.shape
    return np.exp(- np.sum((x - y)**2 / h**2 / 2))

#############################
### better pairwise distances
@jit
def squared_distance_matrix(x):
    """
    Parameters:
    * x: np array of shape (n, d) or (n,)
    Returns:
    * np array of shape (n, n):
    consisting of squared distances ||xi - xj||^2
    """
    n = x.shape[0]
    if x.ndim == 1:
        x = np.reshape(x, (n, 1)) # add dummy dimension
    xx = np.tile(x, (n, 1, 1)) # shape (n, n, d)
    diff = xx - xx.transpose((1, 0, 2))
    return vv_normsq(diff)

def getn(l):
    """
    IN: l = n^2 - n / 2
    OUT: n (positive integer solution)
    """
    n = (1 + np.sqrt(1 + 8*l)) / 2
    assert np.equal(np.mod(n, 1), 0) # make sure n is an integer
    assert l == n**2 - n / 2 # is this assertion too strong? cause n is float
    assert l < n**2 - n / 2 + 0.1 and l > n**2 - n / 2 - 0.1 # just to be safe
    return int(n)

@jit
def squareform(distances):
    """
    IN: output from `pairwise_distances`, an array of length l = n^2 - n / 2 with entries d(x1, x2, d(x1, 3), ..., d(xn-1 xn)).
    OUT: a symmetric n x n distance matrix with entries d(x_i, x_j)
    """
    l = distances.shape[0]
    n = getn(l)
    out = np.zeros((n, n))
    out[np.triu_indices(n, k = 1)]

    out = index_update(out, index[np.triu_indices(n, k=1)], distances)
    out = out + out.T
    return out


#########################33
### Gaussian mixture pdf
@jit
def log_gaussian_mixture(x, means, variances, weights):
    """
    IN:
    * x: scalar
    * mean: np array of means
    * variances: np array
    * weights: np array of weights, same length as the mean and varance arrays

    OUT:
    scalar, value of log(p(x)), where p is the mixture pdf.
    """
    x = np.array(x)
    means, variances, weights = np.array([means, variances, weights])
    exponents = - (x - means)**2 / 2
    c = np.min(exponents)
    norm_consts = 1 / np.sqrt(2 * np.pi * variances) # alternatively, leave this out (not
                                                     # necessary, no need to normalize)

    # normalize smallest element to 0 to avoid underflow errors
    weights = weights * norm_consts
    expout = np.vdot(weights, np.exp(exponents - c))
    out = np.log(expout) + c

    return np.squeeze(out)

## multidim standard gaussian pdf
from jax.scipy.stats import norm

@jit
def standard_normal_logpdf(x):
    """
    Parameters:
    * x: np array of shape (d,)

    Returns:
    * scalar log(p(x)), where p(x) is multidim gaussian
    """
    assert x.ndim == 1
    out = norm.logpdf(x, loc=0, scale=1)
    out = np.prod(out)
    return np.squeeze(out) # to be able to take a gradient, output must be scalar

##########################33
### cartesian product
from jax.ops import index_update, index
@jit
def cartesian_product(*arrays):
    """
    IN: any number of np arrays of same length
    OUT: cartesian product of the arrays
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
#         arr[...,i] = a
        arr = index_update(arr, index[..., i], a)
    return arr.reshape(-1, la)
