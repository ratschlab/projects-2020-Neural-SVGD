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

def phistar_j(x, y, logp, bandwidth):
    """Individual summand needed to compute phi^*. That is, phistar_i = \sum_j phistar_j(xj, xi, logp, bandwidth).
    Arguments:
    * x, y: np.array of shape (d,) or scalar (if d=1)
    * logp: callable
    * bandwidth: scalar or np.array of shape (d,)
    Returns: np.array of shape (d,) or scalar; same shape as x and y
    """
    if x.shape != y.shape:
        raise ValueError("Shapes of particles x and y need to match.")
    kernel = lambda x, y: utils.ard(x, y, bandwidth)
    return grad(logp)(x) * kernel(x, y) + grad(kernel)(x, y)

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
    else:
        pass
    k = lambda y: utils.ard(y, xi, bandwidth)
    return stein.stein(k, x, logp)

def _phistar_i(xi, x, logp, bandwidth):
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
    xi_rep = np.repeat(xi, n, axis=0)
    return np.sum(vmap(phistar_j, (0, 0, None, None))(x, xi_rep, logp, bandwidth), axis=0) # Original state. TODO check what is true and what not.

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
    """
    if x.ndim == 2:
        return vmap(median_heuristic, 1)(x)
    elif x.ndim == 1:
        n = x.shape[0]
        medsq = np.median(utils.squared_distance_matrix(x))
        h = np.sqrt(medsq / np.log(n) / 2)
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
                self.get_bandwidth = median_heuristic
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
        return x, log

    svgd = utils.verbose_jit(svgd, static_argnums=0)

    def step(self, rkey, x, logh, lr, svgd_stepsize, ksd_logh=1, gradient_clip_threshold=100):
        """SGD update step
        1) compute SVGD step (forward)
        2) compute Loss (backward)
        3) update bandwidth via SGD"""
        if ksd_logh is None:
            ksd_logh = logh

        def loss(logh): # TODO add ksd_logh as argument, or make sure jax works like this
            bandwidth = np.exp(logh)
            ksd_bandwidth = np.exp(ksd_logh)
            xout = update(x, self.logp, svgd_stepsize, bandwidth, None, False, None)
            xout = xout + random.normal(rkey, shape=x.shape) * self.noise
            return metrics.ksd(xout, self.logp, ksd_bandwidth)

        current_loss = loss(logh)
        gradient = jacfwd(loss)(logh)
        gradient = gradient / np.linalg.norm(gradient) # normalize

        log = {
            "loss": current_loss,
        }
        updated_logh = logh - lr * optimizers.clip_grads(gradient, gradient_clip_threshold)
        return updated_logh, log

    step = utils.verbose_jit(step, static_argnums=0)

    def train(self, rkey, bandwidth, lr, svgd_stepsize, n_steps, ksd_bandwidth=None, update_after=25):
        bandwidth = np.array(bandwidth, dtype=np.float32) # cast to float so it works with jacfwd
        logh = np.log(bandwidth)

        x = self.initialize(rkey)
        log = metrics.initialize_log(self)
        log["x0"] = x
        loss = []
        if ksd_bandwidth is None:
            concurrent = True
        else:
            ksd_logh = np.log(ksd_bandwidth)
            concurrent = False

        historical_grad = np.zeros(shape=self.particle_shape)
        for i in tqdm(range(n_steps)):
            update_logh = i > update_after

            if update_logh:
                # take "dream" step, compute loss and update logh
                if concurrent:
                    ksd_logh=logh

                rkey = random.split(rkey)[0]
                logh, loss_log = self.step(rkey, x, logh, lr, svgd_stepsize, ksd_logh)
                loss.append(loss_log["loss"])

            bandwidth = np.exp(logh)
            log = metrics.update_log(self, i, x, log, bandwidth)

            x = update(x, self.logp, svgd_stepsize, bandwidth, x, self.adagrad, historical_grad)
            rkey = random.split(rkey)[0]
            x = x + random.normal(rkey, shape=x.shape) * self.noise

            if np.any(np.isnan(logh)):
                warnings.warn(f"NaNs detected in logh at iteration {i}. Training interrupted.", RuntimeWarning)
                break
            elif np.any(np.isnan(x)):
                warnings.warn(f"NaNs detected in x at iteration {i}. Logh is fine, which means NaNs come from update. Training interrupted.", RuntimeWarning)
                break
        return x, log, loss #x_grad, x, log_grad, log_svgd, loss





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
#        x0 = random.normal(rkey, shape=(100, 1)) - 10
        x = x0
        rkey = random.split(rkey)[0]
        xest = self.initialize(rkey)

        log = metrics.initialize_log(self)

        def update_fun(i, u):
            """Compute updated particles and log metrics."""
            x, xest, log, rkey = u
            if self.adaptive_kernel:
                _bandwidth = self.get_bandwidth(x)
            else:
                _bandwidth = bandwidth

            log = metrics.update_log(self, i, x, log, _bandwidth)
            x    = update(x,    self.logp, stepsize, _bandwidth, xest, False, None)
            xest = update(xest, self.logp, stepsize, _bandwidth, xest, False, None)

            rkey = random.split(rkey)[0]
#            eps = 10e-2
#            xest, x = [xs + random.normal(rkey, shape=xs.shape) * eps for xs in (xest, x)]
            return [x, xest, log, rkey]
        init = [x, xest, log, rkey]
        x, xest, log, _ = lax.fori_loop(0, self.n_iter_max, update_fun, init)
        log["x0"] = x0
        return x, xest, log

    svgd_twotrack = utils.verbose_jit(svgd_twotrack, static_argnums=0)
