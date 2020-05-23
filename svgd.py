import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd
from jax import lax
from jax.ops import index, index_add, index_update
from jax.experimental import optimizers

import time
from tqdm import tqdm
import warnings

import utils
import metrics
import stein

def phistar_i(xi, x, logp, bandwidth):
    """
    Arguments:
    * xi: np.array of shape (d,), usually a row element of x
    * x: np.array of shape (n, d)
    * logp: callable
    * bandwidth: scalar or np.array of shape (d,)

    Returns:
    * \phi^*(xi) estimated using the particles x
    """
    if xi.ndim > 1:
        raise ValueError(f"Shape of xi must be (d,). Instead, received shape {xi.shape}")

    k = lambda y: utils.ard(y, xi, bandwidth)
    return stein.stein(k, x, logp)

def phistar(x, logp, bandwidth, xest=None):
    """
    Returns an np.array of shape (n, d) containing values of phi^*(x_i) for i in {1, ..., n}.
    Optionally, supply an array xest of the same shape. This array will be used to estimate phistar; the trafo will then be applied to x.
    """
    if xest is None:
        return vmap(phistar_i, (0, None, None, None))(x, x, logp, bandwidth)
    else:
        return vmap(phistar_i, (0, None, None, None))(x, xest, logp, bandwidth)

# adagrad params (fixed):
alpha = 0.9
fudge_factor = 1e-6
def update(x, logp, stepsize, bandwidth, xest=None, adagrad=False, historical_grad=None):
    """SVGD update step / forward pass
    If xest is passed, update is computed using particles xest, but is still applied to x.
    """
    if adagrad:
        phi = phistar(x, logp, bandwidth, xest)
        historical_grad = alpha * historical_grad + (1 - alpha) * (phi ** 2)
        phi_adj = np.divide(phi, fudge_factor+np.sqrt(historical_grad))
        x = x + stepsize * phi_adj
    else:
        x = x + stepsize * phistar(x, logp, bandwidth, xest)
    return x
update = jit(update, static_argnums=(1, 5))

def median_heuristic(x):
    """
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

class SVGD():
    def __init__(self, dist, n_iter_max, adaptive_kernel=False, get_bandwidth=None, particle_shape=None, adagrad=False, noise=0):
        """
        Arguments:
        * dist: instance of class metrics.Distribution. Alternatively, a callable that computes log(p(x))
        * n_iter_max: integer
        * adaptive_kernel: bool
        * get_bandwidth: callable or None
        * noise: scalar. level of gaussian noise added at each iteration.
        """
        if adaptive_kernel:
            if get_bandwidth is None:
                self.get_bandwidth = lambda x: np.log(median_heuristic(x))
        else:
            assert get_bandwidth is None

        # changing these triggers recompile of method svgd
        if isinstance(dist, metrics.Distribution):
            self.logp = dist.logpdf
            self.dist = dist
            if particle_shape is None:
                self.particle_shape = (100, dist.d) # default nr of particles
            else:
                if particle_shape[1] != dist.d: raise ValueError(f"Distribution is defined on R^{dist.d}, but particle dim was given as {particle_shape[1]}. If particle_shape is given, these values need to be equal.")
                self.particle_shape = particle_shape
        elif callable(dist):
            self.logp = dist
            warnings.warn("No dist instance supplied. This means you won't have access to the SVGD.dist.compute_metrics method, which computes the metrics logged during calls to SVGD.svgd and SVGD.step.")
            self.dist = None
        else:
            raise ValueError()
        self.n_iter_max = n_iter_max
        self.adaptive_kernel = adaptive_kernel
        self.adagrad = adagrad

        # these don't trigger recompilation (never used as context in jitted functions)
        self.rkey = random.PRNGKey(0)

        # these don't trigger recompilation, but also the compiled method doesn't noticed they changed.
        self.ksd_kernel_range = np.array([0.1, 1, 10])
        self.noise = noise

    def newkey(self):
        """Not pure"""
        self.rkey = random.split(self.rkey)[0]

    def initialize(self, rkey=None):
        """Initialize particles distributed as N(-10, 1)."""
        if rkey is None:
            rkey = self.rkey
            self.newkey()
        return random.normal(rkey, shape=self.particle_shape) - 5

    def svgd(self, x0, stepsize, bandwidth, n_iter):
        """
        IN:
        * rkey: random seed
        * stepsize is a float
        * bandwidth is an np array of length d: bandwidth parameter for RBF kernel
        * n_iter: integer, has to be less than self.n_iter_max
        * n: integer, number of particles

        OUT:
        * Updated particles x (np array of shape n x d) after self.n_iter steps of SVGD
        * dictionary with logs
        """
        x = x0
        log = metrics.initialize_log(self)
        log["x0"] = x0
        particle_shape = x.shape
        historical_grad = np.zeros(shape=particle_shape)
        rkey = random.PRNGKey(7)
        def update_fun(i, u):
            """Compute updated particles and log metrics."""
            x, log, historical_grad, rkey = u
            if self.adaptive_kernel:
                _bandwidth = self.get_bandwidth(x)
            else:
                _bandwidth = bandwidth

            log = metrics.update_log(self, i, x, log, _bandwidth)
            x = update(x, self.logp, stepsize, _bandwidth, None, self.adagrad, historical_grad)

            rkey = random.split(rkey)[0]
            x = x + random.normal(rkey, shape=x.shape) * self.noise

            return [x, log, historical_grad, rkey]

        x, log, *_ = lax.fori_loop(0, self.n_iter_max, update_fun, [x, log, historical_grad, rkey])
        log["metric_names"] = self.dist.metric_names
        return x, log
    svgd = utils.verbose_jit(svgd, static_argnums=0)

    def kernel_step(self, h, x, stepsize):
        """Update h <- h + \nabla KSD_{h}(q, p)
        Arguments:
        * x: np.array of shape (n,d).
        * h: scalar or np.array of shape (d,)
        * stepsize: scalar"""
        def ksd_sq(h):
            return metrics.ksd_squared(x, self.logp, h)
        return h + stepsize * optimizers.clip_grads(grad(ksd_sq)(h), 0.00005)
    kernel_step = jit(kernel_step, static_argnums=0)

    def svgd_step(self, x, h, stepsize):
        """Update X <- X + \phi^*_{p,q}(X)
        Arguments:
        * x: np.array of shape (n,d).
        * h: scalar or np.array of shape (d,)
        * stepsize: scalar"""
        return x + stepsize * phistar(x, self.logp, h)
    svgd_step = jit(svgd_step, static_argnums=0)

    def train(self, rkey, bandwidth, lr, svgd_stepsize, n_steps):
        bandwidth = np.array(bandwidth, dtype=np.float32) # cast to float so it works with jacfwd
        logh = np.log(bandwidth) # working with log(bandwidth) is more robust.

        x = self.initialize(rkey)
        log = metrics.initialize_log(self)
        log["x0"] = x
        ksd_pre = []
        ksd_post = []
        for i in tqdm(range(n_steps)):
            ksd_pre.append(metrics.ksd_squared(x, self.logp, bandwidth))
            bandwidth = self.kernel_step(bandwidth, x, lr)
            ksd_post.append(metrics.ksd_squared(x, self.logp, bandwidth))
            x = self.svgd_step(x, bandwidth, svgd_stepsize)
            log = metrics.update_log(self, i, x, log, bandwidth)

            if np.any(np.isnan(bandwidth)):
                warnings.warn(f"NaNs detected in bandwidth at iteration {i}. Training interrupted.", RuntimeWarning)
                break
            elif np.any(np.isnan(x)):
                warnings.warn(f"NaNs detected in x at iteration {i}. Logh is fine, which means NaNs come from update. Training interrupted.", RuntimeWarning)
                break
        log["metric_names"] = self.dist.metric_names
        log["ksd_pre"] = np.array(ksd_pre)
        log["ksd_post"] = np.array(ksd_post)
        return x, log
