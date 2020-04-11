import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd
from jax.lax import fori_loop
from utils import ard, squared_distance_matrix
from jax.ops import index, index_add, index_update


def ard_matrix(x, bandwidth):
    """
    Arguments:
    * x, np array of shape (n, d)
    * kernel bandwidth, np array of shape (d,) or one-dimensional float

    Returns:
    * np array of shape (n, n) containing values k(xi, xj) for xi = x[i, :].
    """
    bandwidth = np.array(bandwidth)
    dsquared = vmap(squared_distance_matrix, 1)(x) # shape (d, n, n)
    if bandwidth.ndim > 0 and bandwidth.shape[0] > 1:
        bandwidth = bandwidth[:, np.newaxis, np.newaxis] # reshape bandwidth to have same shape as dsquared
    return np.exp(np.sum(- dsquared / bandwidth**2 / 2, axis=0)) # shape (n, n)

def update(x, logp, stepsize, bandwidth):
    km = lambda x: ard_matrix(x, bandwidth)
    kxy = km(x)
    dkxy = jacfwd(km)(x) # (n, n, n, d)
    dkxy = dkxy.diagonal(axis1=1, axis2=2) # (n, d, n)
    dlogp = vmap(grad(logp))(x)

    return x + stepsize * (np.einsum("il,ij->jl", dlogp, kxy) + np.sum(dkxy, axis=2))

def get_bandwidth(x):
    """
    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: Updated bandwidth parameter for RBF kernel, based on update rule from the SVGD paper.
    """
    if x.ndim == 2:
        return vmap(get_bandwidth, 1)(x)
    elif x.ndim == 1:
        n = x.shape[0]
        medsq = np.median(squared_distance_matrix(x))
        h = np.sqrt(medsq / np.log(n) / 2)
        return h
    else:
        raise ValueError("Shape of x has to be either (n,) or (n, d)")


class SVGD():
    def __init__(self, logp, n_iter_max, adaptive_kernel=False, get_bandwidth=None):
        if adaptive_kernel:
            assert get_bandwidth is not None
        else:
            assert get_bandwidth is None

        self.logp = logp
        self.n_iter_max = n_iter_max
        self.adaptive_kernel = adaptive_kernel
        self.get_bandwidth = get_bandwidth

    def unjitted_svgd(self, x, stepsize, bandwidth, n_iter):
        """
        IN:
        * x is an np array of shape n x d (n particles of dimension d)
        * stepsize is a float
        * bandwidth is an np array of length d: bandwidth parameter for RBF kernel

        OUT:
        * Updated particles x (np array of shape n x d) after self.n_iter_max steps of SVGD
        * dictionary with logs
        """
        assert x.ndim == 2
        d = x.shape[1]
        log = {
            "particle_mean": np.zeros(shape=(self.n_iter_max, d)),
            "particle_var":  np.zeros(shape=(self.n_iter_max, d))
        }

        if self.adaptive_kernel:
            log["bandwidth"] = np.zeros(shape=(self.n_iter_max, d))

        def update_fun(i, u):
            """
            1) if self.adaptive_kernel, compute bandwidth from x
            2) compute updated x,
            3) log mean and var (and bandwidth)

            Parameters:
            * i: iteration counter (used to update log)
            * u = [x, log]

            Returns:
            [updated_x, log]
            """
            x, log = u
            if self.adaptive_kernel:
                adaptive_bandwidth = get_bandwidth(x)
                x = update(x, self.logp, stepsize, adaptive_bandwidth)
            else:
                x = update(x, self.logp, stepsize, bandwidth)

            update_dict = {
                "particle_mean": np.mean(x, axis=0),
                "particle_var": np.var(x, axis=0)
            }
            if self.adaptive_kernel:
                update_dict["bandwidth"] = adaptive_bandwidth

            for key in log.keys():
                log[key] = index_update(log[key], index[i, :], update_dict[key])

            return [x, log]

        x, log = fori_loop(0, n_iter, update_fun, [x, log]) # when I wanna do grad(svgd), I need to reimplement fori_loop using scan (which is differentiable).

#        for k, v in log.items():
#            log[k] = v[:n_iter]
        return x, log

    # this way we only print "COMPILING" when we're actually compiling
    def svgd(self, x, stepsize, bandwidth, n_iter):
        print("COMPILING")
        return self.unjitted_svgd(x, stepsize,bandwidth, n_iter)
    svgd = jit(svgd, static_argnums=0)
