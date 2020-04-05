import jax.numpy as np
from jax import grad, jit, vmap, random
from jax.lax import fori_loop
from utils import ard, squared_distance_matrix
from jax.ops import index, index_add, index_update

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

def update(x, logp, stepsize, bandwidth):
    """
    IN:
    * x: np array of shape n x d
    * logp: callable, log of a differentiable pdf p
    * stepsize: scalar > 0
    * bandwidth: np array (or dict?) fed to kernel(x, y, bandwidth)

    OUT:
    xnew = x + stepsize * \phi^*(x)
    that is, xnew is an array of shape n x d. The entries of x are the updated particles.

    note that this is an inefficient way to do things, since we're computing k(x, y) twice for each x, y combination.
    """
    assert x.ndim == 2
    kernel = lambda x, y: ard(x, y, bandwidth)

    xnew = []
    n = x.shape[0]
    for i, xi in enumerate(x):
        repeated = np.tile(xi, (n, 1))
        xnew.append(stepsize * np.sum(phi_j_batched(x, repeated, logp, kernel), axis = 0))
    xnew = np.array(xnew)
    xnew += x

    return xnew

# update = jit(update, static_argnums=(1,)) # logp is static. When logp changes, jit recompiles.

# @jit
def get_bandwidth(x):
    """
    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: Updated bandwidth parameter for RBF kernel, based on update rule from the SVGD paper.
    """
    if x.ndim == 2:
        return vmap(get_bandwidth, 1)(x)
    elif x.ndim == 1:
        n = x.shape[0]
        h = np.median(squared_distance_matrix(x)) / np.log(n)
        return h
    else:
        raise ValueError("Shape of x has to be either (n,) or (n, d)")


class SVGD():
    def __init__(self, logp, n_iter, adaptive_kernel=False, get_bandwidth=None):
        if adaptive_kernel:
            assert get_bandwidth is not None
        else:
            assert get_bandwidth is None

        self.logp = logp
        self.n_iter = n_iter
        self.adaptive_kernel = adaptive_kernel
        self.get_bandwidth = get_bandwidth

    def svgd(self, x, stepsize, bandwidth=1):
        """
        IN:
        * x is an np array of shape n x d
        * stepsize is a float
        * bandwidth is a positive scalar: bandwidth parameter for RBF kernel

        OUT:
        * Updated particles x (np array of shape n x d) after self.n_iter steps of SVGD
        * dictionary with logs
        """
        assert x.ndim == 2

        d = x.shape[1]
        log = {
            "particle_mean": np.empty(shape=(self.n_iter, d)),
            "particle_var":  np.empty(shape=(self.n_iter, d))
        }

        if self.adaptive_kernel:
            log["bandwidth"] = np.empty(shape=(self.n_iter, d))

        def update_fun(i, u):
            """
            1) if adaptive_kernel, compute bandwidth from x
            2) compute updated x,
            3) log mean and var (and bandwidth)

            Parameters:
            * i: iteration counter (unused)
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

        x, log = fori_loop(0, self.n_iter, update_fun, [x, log]) # when I wanna do grad(svgd), I need to reimplement fori_loop using scan (which is differentiable).

        return x, log

    svgd = jit(svgd, static_argnums=0)

#    def adaptive_param_svgd(self, x, logp, stepsize, L):
#        """
#        IN:
#        * x is an np array of shape n x d
#        * logp is the log of a differentiable pdf p (callable)
#        * stepsize is a float
#        * L is an integer (number of iterations)
#
#        OUT:
#        * Updated particles x (np array of shape n x d) after L steps of SVGD
#        * dictionary with logs
#        """
#        assert x.ndim == 2
#
#        d = x.shape[1]
#        log = {
#            "kernel_param": np.empty(shape=(L, d)),
#            "particle_mean": np.empty(shape=(L, d)),
#            "particle_var": np.empty(shape=(L, d))
#        }
#
#        def update_fun(i, u):
#            """
#            1) compute kernel_param from x
#            2) compute updated x,
#            3) log everything
#
#            Parameters:
#            * i: iteration counter (unused)
#            * u = [x, log]
#            """
#            x, log = u
#            kernel_param = get_bandwidth(x)
#            x = update(x, logp, stepsize, kernel_param)
#
#            update_dict = {
#                "kernel_param": kernel_param,
#                "particle_mean": np.mean(x, axis=0),
#                "particle_var": np.var(x, axis=0)
#            }
#
#            for key in log.keys():
#                log[key] = index_update(log[key], index[i, :], update_dict[key])
#
#            return [x, log]
#
#        x, log = fori_loop(0, L, update_fun, [x, log])
#
#        return x, log
#
#    adaptive_param_svgd = jit(adaptive_param_svgd, static_argnums=(0, 2, 4))


