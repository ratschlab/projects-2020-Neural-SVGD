import jax.numpy as np
from jax import grad, jit, vmap, random
from jax.ops import index_update, index
from jax.scipy.special import logsumexp
from jax import lax

def clip(gradient, threshold):
    r = np.linalg.norm(gradient)
    if r > threshold:
        return gradient * threshold / r
    else:
        return gradient

def is_pd(x):
    """check if matrix is positive defininite"""
    import numpy as onp
    try:
        onp.linalg.cholesky(x)
        return True
    except onp.linalg.linalg.LinAlgError as err:
        return False

## fori_loop implementation in terms of lax.scan taken from here https://github.com/google/jax/issues/1112
def fori_loop(lower, upper, body_fun, init_val):
    f = lambda x, i: (body_fun(i, x), ())
    result, _ = lax.scan(f, init_val, np.arange(lower, upper))
    return result

# this one from here https://github.com/google/jax/issues/650
# def fori_loop(_, num_iters, fun, init): # added the dummy _
#     dummy_inputs = np.zeros((num_iters, 0))
#     out, _ = lax.scan(lambda x, dummy: (fun(x), dummy), init, dummy_inputs)
#     return out

# this is the python equivalent given in the documentation https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html
def python_fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val






############ Kernel
# @jit
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
    x, y = np.array([x, y])
    assert x.ndim == 1
    assert y.ndim == 1
    return np.exp(- normsq(x - y) / (2 * h))

# @jit
def ard(x, y, h):
    """
    IN:
    * x, y: np arrays of shape (d,)
    * h: np array of shape (d,), or scalar

    OUT:
    scalar kernel(x, y, h).
    """
    x, y = np.array([x, y])
    assert x.ndim <= 1
    assert y.ndim <= 1

    h = np.array(h)
    assert h.ndim == 1 or h.ndim == 0
    assert x.shape == y.shape
    if h.ndim == 1:
        assert x.shape == h.shape
    return np.exp(- np.sum((x - y)**2 / h**2) / 2)

#############################
### better pairwise distances
# @jit
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

    n = int(n)
    assert l == n**2 - n / 2
    return n

# @jit
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
# @jit
def log_gaussian_mixture(x, means, variances, weights):
    """
    IN:
    * x: scalar or array of length n
    * mean: np array of means
    * variances: np array
    * weights: np array of weights, same length as the mean and varance arrays

    OUT:
    scalar, value of log(p(x)), where p is the mixture pdf.
    """
    x = np.array(x)
    if x.ndim == 2:
        assert x.shape[1] == 1
    else:
        assert x.ndim <= 1
    means, variances, weights = np.array([means, variances, weights])
    exponents = - (x - means)**2 / 2
    norm_consts = 1 / np.sqrt(2 * np.pi * variances) # alternatively, leave this out (not
                                                     # necessary, no need to normalize)
    weights = weights * norm_consts
    exponents = exponents + np.log(weights)
    out = logsumexp(exponents)

    return np.squeeze(out)

## multidim standard gaussian pdf
from jax.scipy.stats import norm

# @jit
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
# @jit
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

def dict_concatenate(dict_list):
    """
    Arguments:
    * dict_list: a list of dictionaries with the same keys. All values must be numeric.

    Returns:
    * a dictionary with the same keys as the input dictionaries. The values are np
    arrays consisting of the concatenation of the values in the input dictionaries.
    """
    assert all([dict_list[i].keys() == dict_list[i+1].keys() for i in range(len(dict_list)-1)])

    keys = dict_list[0].keys()
    out = {key: [d[key] for d in dict_list] for key in keys}

    for k, v in out.items():
        out[k] = np.array(v)

    return out

def dict_mean(dict_list):
    """
    Arguments:
    * dict_list: a list of dictionaries with the same keys. All values must be numeric.

    Returns:
    * a dictionary with the same keys as the input dictionaries. The values are np
    arrays consisting of the mean of the values in the input dictionaries.
    """
    out = dict_concatenate(dict_list)
    for k, v in out.items():
        out[k] = np.mean(v, axis = 0)
        assert out[k].shape == dict_list[0][k].shape

    return out


#########################
## distributions
