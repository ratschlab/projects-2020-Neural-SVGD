import jax.numpy as np
import jax
from jax import grad, jit, vmap, random
from jax.ops import index_update, index
from jax.scipy.special import logsumexp
from jax import lax
import time
from tqdm import tqdm
from functools import wraps

##############################
### KL divergence utilities
def smooth_and_normalize(vec, normalize=True):
    """
    Parameters:
    * vec : np.array of shape (n,)
    * normalize : bool

    Returns:
    out : np.array of shape (n,).
    If vec_i = 0, then out_i = epsilon. If vec_i !=0, then out_i = vec_i - c.
    c is chosen such that sum(vec) == 1.
    """
    vec = np.asarray(vec, dtype=np.float32)

    if normalize:
        vec = vec / vec.sum()
    n = len(vec)
    epsilon = 0.0001
    num_nonzero = np.count_nonzero(vec)
    c = epsilon * (n - num_nonzero) / num_nonzero
    perturbation =  (vec == 0)*epsilon - (vec != 0)*c
    return vec + perturbation

def get_bins_and_bincounts(samples, normalized=False):
    """take in samples, create a common set of bins, and compute the counts count(x in bin)
    for each bin and each sample x.
    Parameters
    ------------
    samples : np.array of shape (n,) or shape (k, n).
    - If shape (n,): interpreted as a set of n scalar-valued samples.
    - If shape (k, n): interpreted as k sets of n scalar-valued samples.

    Returns
    --------
    probabilities :
    bins :
    """
    nr_samples = np.prod(samples.shape)
    nr_bins = np.log2(nr_samples)
    nr_bins = int(max(nr_bins, 5))

    lims = [np.min(samples), np.max(samples)]
    bins = np.linspace(*lims, num=nr_bins)

    if samples.ndim == 2:
        out = np.asarray([np.histogram(x, bins=bins, density=normalized)[0] for x in samples])
        return out, bins
    elif samples.ndim == 1:
        return np.histogram(samples, bins=bins, density=normalized)[0], bins
    else:
        raise ValueError(f"Input must have shape (n,) or shape (k,n). Instead received shape {samples.shape}")

def get_histogram_likelihoods(samples):
    """
    Parameters:
    * samples : np.array of scalar-valued samples from a distribution.

    Returns:
    np.array of same length as samples, consisting of a histogram-based approximation of the pdf q(x_i) at the samples x_i
    """
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.squeeze(samples)
    if samples.ndim != 1:
        raise ValueError(f"The shape of samples has to be either (n,) or (n,1). Instead received shape {samples.shape}.")
    n = len(samples)

    bincounts, bins = get_bins_and_bincounts(samples)
    bincounts = np.array(bincounts, dtype=np.int32)
    likelihoods = smooth_and_normalize(bincounts) / np.diff(bins)

    sample_likelihoods = np.repeat(likelihoods, bincounts) # TODO this doesn't play well with jit, cause shape of output depends on values in bincounts
    return sample_likelihoods

##############
### sweep
def sweep(rkey, grid, sample_each_time=False, joint_param=False, average_over=1):
    """Sweep a grid of bandwidth values and output corresponding metrics.
    Arguments:
    * average_over: integer, compute average over m random seeds
    * metrics: callable, takes bandwidth h as only input and oututs a either a scalar or an np.array of shape (k,)
    """
    if average_over == 1:
        sweep_results = []
        if sample_each_time:
            rkeys = random.split(svgd_fix.rkey, len(grid))
        else:
            rkeys = [rkey] * len(grid)

        for rkey, h in tqdm(zip(rkeys, grid)):
            if joint_param:
                res = metrics(h)
            else:
                res = metrics(h)
            sweep_results.append(res)

        sweep_results = np.array(sweep_results)
    else:
        ress = []
        for _ in range(average_over):
            rkey = random.split(rkey)[0]
            res = sweep(rkey, grid, sample_each_time=False, joint_param=joint_param, average_over=1)
            ress.append(res)
        ress = np.array(ress)
        sweep_results = np.mean(ress, axis=0)
    if np.any(np.isnan(sweep_results)): print("NaNs detected!")
    return sweep_results

# wrapper that prints when the function compiles
def verbose_jit(fun, *jargs, **jkwargs):
    """Does same thing as jax.jit, only that it also inserts a print statement."""
    @wraps(fun)
    def verbose_fun(*args, **kwargs):
        print(f"JIT COMPILING {fun.__name__}...")
        st = time.time()
        out = fun(*args, **kwargs)
        end = time.time()
        print(f"...done compiling {fun.__name__} after {end-st} seconds.")
        return out
    return jit(verbose_fun, *jargs, **jkwargs)

def check_for_nans(thing):
    """not supposed to end up inside jit ofc"""
    if type(thing) is jax.interpreters.xla.DeviceArray:
        if np.isnan(thing):
            warnings.warn("Detected NaNs in output.", RuntimeWarning)
    elif type(thing) is tuple or type(thing) is list:
        for el in thing:
            check_for_nans(el)
    elif type(thing) is dict:
        for k, v in thing.items():
            check_for_nans(v)
    else:
        warnings.warn("Didn't recognize type. Not checking for NaNs.", RuntimeWarning)
    return None

# wrapper that checks output for NaNs and returns a warning if isnan
def warn_if_nan(fun):
    @wraps(fun)
    def checked_fun(*args, **kwargs):
        out = fun(*args, **kwargs)
        check_for_nans(out)
        return out
    return checked_fun

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

#############################
### better pairwise distances

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

    def normsq(x):
        assert x.ndim == 1
        x = np.array(x)
        return np.vdot(x, x)
    v_normsq = vmap(normsq) # outputs vector of norms
    vv_normsq = vmap(v_normsq)

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
    * dict_list: a list of dictionaries with the same keys. All values must be numeric or a nested dict.

    Returns:
    * a dictionary with the same keys as the input dictionaries. The values are np
    arrays consisting of the concatenation of the values in the input dictionaries.
    """
    for d in dict_list:
        if type(d) is not dict:
            raise TypeError("Input has to be a list consisting of dictionaries.")
        elif not all([dict_list[i].keys() == dict_list[i+1].keys() for i in range(len(dict_list)-1)]):
            raise ValueError("The keys of all input dictionaries need to match.")

    keys = dict_list[0].keys()
    out = {key: [d[key] for d in dict_list] for key in keys}

    for k, v in out.items():
        try:
            out[k] = np.array(v)
        except TypeError:
            out[k] = dict_concatenate(v)

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
        try:
            out[k] = np.mean(v, axis = 0)
            assert out[k].shape == dict_list[0][k].shape
        except TypeError:
            out[k] = dict_mean(v)
    return out

def dict_divide(da, db):
    """divide numeric dict recursively, a / b."""
    for (k, a), (k, b) in zip(da.items(), db.items()):
        try:
            da[k] = a / b
        except TypeError:
            da[k] = dict_divide(a, b)
    return da
