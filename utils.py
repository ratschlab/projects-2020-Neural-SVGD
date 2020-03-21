import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.ops import index_update, index

############ Kernel
@jit
def normsq(x):
    x = np.array(x)
    return np.vdot(x, x)

@jit
def single_rbf(x, y, h):
    """
    x and y are d-dimensional arrays
    h is a scalar parameter
    """
    assert x.ndim == 1 and y.ndim == 1
    return np.exp(- normsq(x - y) / (2 * h**2))

batched_rbf = vmap(single_rbf, (0, 0, None), 0)
batched_normsq = vmap(normsq) # outputs vector of diffs

### distances, norm

#############################
### better pairwise distances
@jit
def pairwise_distances(x):
    """
    IN: n x d array: n observations of d-dimensional samples
    OUT: np array of shape (l,) where l = (n^2 - n) / 2
    Consists of squared euclidian distances d(x1, x2)^2, d(x1, x3)^2, ..., d(xn-1, xn)^2
    """
    assert x.ndim == 2
    n = x.shape[0]
    distances = []
    for i, xi in enumerate(x[:-1]):
        repeated = np.tile(xi, (n - i - 1, 1))
        v = batched_normsq(repeated - x[i+1:]) # length n - i - 1
        distances.extend(v)
    return np.array(distances) # norm squared!

def getn(l):
    """
    IN: l = n^2 - n / 2
    OUT: n (positive integer solution)
    """
    n = (1 + np.sqrt(1 + 8*l)) / 2
    assert np.equal(np.mod(n, 1), 0) # make sure n is an integer
    return int(n)

@jit
def get_distance_matrix(distances):
    """
    IN: output from `pairwise_distances`, an array of length l = n^2 - n / 2
    OUT: a symmetric n x n distance matrix with entries d(x_i, x_j)
    """
    l = distances.shape[0]
    n = getn(l)
    out = np.zeros((n, n))
    out[np.triu_indices(n, k = 1)]

    out = index_update(out, index[np.triu_indices(n, k=1)], distances)
    out = out + out.T
    return out
