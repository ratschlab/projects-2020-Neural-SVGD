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

def phistar_j(x, y, logp, bandwidth):
    """Individual summand needed to compute phi^*. That is, phistar_i = \sum_j phistar_j(xj, xi, logp, bandwidth)"""
    kernel = lambda x, y: utils.ard(x, y, bandwidth)
    return grad(logp)(x) * kernel(x, y) + grad(kernel)(x, y)

def phistar_i(xi, x, logp, bandwidth):
    """
    Arguments:
    * xi: np.array of shape (1, d), usually a row element of x
    * x: np.array of shape (n, d)
    * logp: callable
    * bandwidth: scalar or np.array of shape (d,)

    Returns:
    * \phi^*(xi) estimated using the particles x
    """
    if xi.ndim == 1:
        xi = xi[np.newaxis, :]
    elif xi.ndim == 0:
        xi = xi[np.newaxis, np.newaxis]
    else:
        pass
    assert xi.ndim == 2

    n = x.shape[0]
    xi_rep = np.repeat(xi, n,axis=0)
    return np.sum(vmap(phistar_j, (0, 0, None, None))(x, xi_rep, logp, bandwidth), axis=0)

def phistar(x, logp, bandwidth, xest=None):
    """
    Returns an np.array of shape (n, d) containing values of phi^*(x_i) for i in {1, ..., n}.
    Optionally, supply an array xest of the same shape. This array will be used to estimate phistar; the trafo will then be applied to x.
    """
    if xest is None:
        return vmap(phistar_i, (0, None, None, None))(x, x, logp, bandwidth)
    else:
        return vmap(phistar_i, (0, None, None, None))(x, xest, logp, bandwidth)

def update(x, logp, stepsize, bandwidth, xest=None):
    """SVGD update step / forward pass"""
    return x + stepsize * phistar(x, logp, bandwidth, xest)
update = jit(update, static_argnums=1)

def get_bandwidth(x):
    """
    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: Updated bandwidth parameter for RBF kernel, based on update rule from the SVGD paper.
    """
    if x.ndim == 2:
        return vmap(get_bandwidth, 1)(x)
    elif x.ndim == 1:
        n = x.shape[0]
        medsq = np.median(utils.squared_distance_matrix(x))
        h = np.sqrt(medsq / np.log(n) / 2)
        return h
    else:
        raise ValueError("Shape of x has to be either (n,) or (n, d)")

class SVGD():
    def __init__(self, dist, n_iter_max, adaptive_kernel=False, get_bandwidth=None, particle_shape=None):
        """
        Arguments:
        * dist: instance of class metrics.Distribution. Alternatively, a callable that computes log(p(x))
        * n_iter_max: integer
        * adaptive_kernel: bool
        * get_bandwidth: callable or None
        """
        if adaptive_kernel:
            assert get_bandwidth is not None
        else:
            assert get_bandwidth is None

        # changing these triggers recompile of method svgd
        if isinstance(dist, metrics.Distribution):
            self.logp = dist.logpdf
            self.dist = dist
            if particle_shape is None:
                self.particle_shape = (100, dist.d) # default nr of particles
            else:
                if particle_shape[1] != dist.d: raise ValueError()
                self.particle_shape = particle_shape
        elif callable(dist):
            self.logp = dist
            warnings.warn("No dist instance supplied. This means you won't have access to the SVGD.dist.compute_metrics method, which computes the metrics logged during calls to SVGD.svgd and SVGD.step.")
            self.dist = None
        else:
            raise ValueError()
        self.n_iter_max = n_iter_max
        self.adaptive_kernel = adaptive_kernel
        self.get_bandwidth = get_bandwidth

        # these don't trigger recompilation:
        self.rkey = random.PRNGKey(0)

        # these don't trigger recompilation, but also the compiled method doesn't noticed they changed.
        self.ksd_kernel_range = np.array([0.1, 1, 10])

    def newkey(self):
        """Not pure"""
        self.rkey = random.split(self.rkey)[0]

    def initialize(self, rkey):
        """Initialize particles distributed as N(-10, 1)."""
        return random.normal(rkey, shape=self.particle_shape) - 10


    def svgd(self, rkey, stepsize, bandwidth, n_iter):
        """
        IN:
        * rkey: random seed
        * stepsize is a float
        * bandwidth is an np array of length d: bandwidth parameter for RBF kernel
        * n_iter: integer, has to be less than self.n_iter_max

        OUT:
        * Updated particles x (np array of shape n x d) after self.n_iter steps of SVGD
        * dictionary with logs
        """
        x0 = self.initialize(rkey)
        x = x0
        log = metrics.initialize_log(self)

        def update_fun(i, u):
            """Compute updated particles and log metrics."""
            x, log = u
            if self.adaptive_kernel:
                adaptive_bandwidth = get_bandwidth(x)
                log = metrics.update_log(self, i, x, log, adaptive_bandwidth)
                x = update(x, self.logp, stepsize, adaptive_bandwidth)
            else:
                log = metrics.update_log(self, i, x, log, bandwidth)
                x = update(x, self.logp, stepsize, bandwidth)

            return [x, log]

        x, log = lax.fori_loop(0, self.n_iter_max, update_fun, [x, log])
        log["x0"] = x0
        return x, log

    # this way we only print "COMPILING" when we're actually compiling
    svgd = utils.verbose_jit(svgd, static_argnums=0)

    def step(self, x, bandwidth, meta_stepsize, svgd_stepsize, ksd_bandwidth=1, gradient_clip_threshold=100):
        """SGD update step
        1) compute SVGD step (forward)
        2) compute Loss (backward)
        3) update bandwidth via SGD"""
        if ksd_bandwidth is None:
            ksd_bandwidth = bandwidth

        def loss(bandwidth):
            xout = update(x, self.logp, svgd_stepsize, bandwidth)
            return metrics.ksd(xout, self.logp, ksd_bandwidth)

        current_loss = loss(bandwidth)
        gradient = jacfwd(loss)(bandwidth)

        log = {
            "loss": current_loss,
        }
        updated_bandwidth = bandwidth - meta_stepsize * optimizers.clip_grads(gradient, gradient_clip_threshold)
        return updated_bandwidth, log

    step = utils.verbose_jit(step, static_argnums=0)

    def train(self, rkey, bandwidth, meta_stepsize, svgd_stepsize, n_steps):
        bandwidth = np.array(bandwidth, dtype=np.float32) # cast to float so it works with jacfwd
        x = self.initialize(rkey)
        log = metrics.initialize_log(self)
        log["x0"] = x
        log["loss"] = []

        for i in range(n_steps):
            log = metrics.update_log(self, i, x, log, bandwidth)

            # take "dream" step, compute loss and update bandwidth
            bandwidth, steplog = self.step(x, bandwidth, meta_stepsize, svgd_stepsize)
            log["loss"].append(steplog["loss"])

            # using updated bandwidth, take actual step
            x = update(x, self.logp, svgd_stepsize, bandwidth)

        return x, log






    def optimize_bandwidth(self, bandwidth, stepsize, n_steps, ksd_bandwidth=1, sample_every_step=True, gradient_clip_threshold=10):
        """Find optimal bandwidth using SGD.
        Not pure (mutates self.rkey)"""
        if self.adaptive_kernel:
            raise ValueError("This SVGD instance uses an adaptive bandwidth parameter.")
        if ksd_bandwidth is None:
            vary_ksd_bandwidth = True
        else:
            vary_ksd_bandwidth = False

        log = [bandwidth]
        for _ in tqdm(range(n_steps)):
            if sample_every_step:
                self.newkey()
            if vary_ksd_bandwidth:
                ksd_bandwidth = bandwidth
            bandwidth = self.step(self.rkey, bandwidth, stepsize, ksd_bandwidth=ksd_bandwidth, gradient_clip_threshold=gradient_clip_threshold)
            if np.any(np.isnan(bandwidth)):
                raise Exception(f"Gradient is NaN. Last non-NaN value of bandwidth was {log[-1]}")
            elif np.any(bandwidth < 0):
                print(f"Note: some entries are below zero. Bandwidth = {bandwidth}.")
            else:
                pass
            log.append(bandwidth)
        return bandwidth, log




    def svgd_twotrack(self, rkey, stepsize, bandwidth, n_iter):
        """
        IN:
        * rkey: random seed
        * stepsize is a float
        * bandwidth is an np array of length d: bandwidth parameter for RBF kernel
        * n_iter: integer, has to be less than self.n_iter_max

        OUT:
        * Updated particles x (np array of shape n x d) after self.n_iter steps of SVGD
        * dictionary with logs
        """
        x0 = self.initialize(rkey)
        x = x0
        rkey = random.split(rkey)[0]
        xest = self.initialize(rkey)

        log = metrics.initialize_log(self)

        def update_fun(i, u):
            """Compute updated particles and log metrics."""
            x, xest, log = u
            if self.adaptive_kernel:
                adaptive_bandwidth = get_bandwidth(x)
                log = metrics.update_log(self, i, x, log, adaptive_bandwidth)
                x = update(x, self.logp, stepsize, adaptive_bandwidth, xest)
                xest = update(xest, self.logp, stepsize, adaptive_bandwidth)
            else:
                log = metrics.update_log(self, i, x, log, bandwidth)
                x = update(x, self.logp, stepsize, bandwidth, xest)
                xest = update(xest, self.logp, stepsize, bandwidth)

            return [x, xest, log]

        x, xest, log = lax.fori_loop(0, self.n_iter_max, update_fun, [x, xest, log])
        log["x0"] = x0
        return x, xest, log

    svgd_twotrack = utils.verbose_jit(svgd_twotrack, static_argnums=0)
