import jax.numpy as np
from jax import grad, jit, vmap, random
from jax import lax
from jax.ops import index, index_add, index_update
from jax.experimental import optimizers

import time
from tqdm import tqdm
import warnings
from functools import partial

import utils
import metrics
import stein
import kernels

verbose_jit = utils.verbose_jit

def phistar_i(xi, x, logp, kernel):
    """
    Arguments:
    * xi: np.array of shape (d,), usually a row element of x
    * x: np.array of shape (n, d)
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y.

    Returns:
    * \phi^*(xi) estimated using the particles x
    """
    if xi.ndim > 1:
        raise ValueError(f"Shape of xi must be (d,). Instead, received shape {xi.shape}")
    kx = lambda y: kernel(y, xi)
    return stein.stein(kx, x, logp)

def phistar(x, logp, kernel):
    """
    Returns an np.array of shape (n, d) containing values of phi^*(x_i) for i in {1, ..., n}.

    Arguments:
    * x: np.array of shape (n, d)
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y.
    """
    return vmap(phistar_i, (0, None, None, None))(x, x, logp, kernel)

# adagrad params (fixed):
alpha = 0.9
fudge_factor = 1e-6
def update(x, logp, stepsize, kernel, adagrad=False, historical_grad=None):
    """SVGD update step / forward pass
    """
    if adagrad:
        phi = phistar(x, logp, kernel)
        historical_grad = alpha * historical_grad + (1 - alpha) * (phi ** 2)
        phi_adj = np.divide(phi, fudge_factor+np.sqrt(historical_grad))
        x = x + stepsize * phi_adj
    else:
        x = x + stepsize * phistar(x, logp, kernel)
    return x
update = jit(update, static_argnums=(1, 5))

class SVGD():
    def __init__(self, dist, n_iter, nr_particles=None, particle_dim=None, kernel=kernels.ard, snapshot_iter=[10, 20, 30, 50, 100]):
        """
        Arguments:
        * dist: instance of class metrics.Distribution. Alternatively, a callable that computes log(p(x))
        * n_iter: integer, nr of SVGD steps.
        * kernel: callable. Takes in three arguments:
            x, y : array-like, shape (d,).
            kernel_params : pytree (ie scalar, or recursive list, dict, or tuple) containing kernel parameters.

        * noise: scalar. level of gaussian noise added at each iteration.
        """
        # check arguments:
        if not callable(kernel):
            raise ValueError("kernel must be callable.")

        if nr_particles is None:
            self.nr_particles = 100 # default nr of particles
        else:
            self.nr_particles = nr_particles

        self.kernel = lambda kernel_params: lambda x, y: kernel(x, y, kernel_params)

        if isinstance(dist, metrics.Distribution):
            self.logp = dist.logpdf
            self.dist = dist
            if dist.d != particle_dim and particle_dim is not None:
                raise ValueError(f"Arguments dist.d and particle_dim need to be equal.\
                                  Instead received {dist.d} and {particle_dim}, respectively.")
            self.d = dist.d
        elif callable(dist):
            self.logp = dist
            warnings.warn("No dist instance supplied. This means you won't have access to\
                          the SVGD.dist.compute_metrics method, which computes the metrics\
                          logged during calls to SVGD.svgd and SVGD.step.")
            self.d = particle_dim
            self.dist = None
        else:
            raise ValueError()

        self.n_iter = n_iter

        # these don't trigger recompilation (never used as context in jitted functions)
        self.rkey = random.PRNGKey(0)

        # these don't trigger recompilation, but also the compiled method doesn't noticed they changed.
        self.ksd_kernel_range = np.array([0.1, 1, 10])
        self.noise = 1e-3
        self.snapshot_iter = snapshot_iter

    def newkey(self):
        """Not pure"""
        self.rkey = random.split(self.rkey)[0]

    def initialize(self, rkey=None):
        """Initialize particles distributed as N(-10, 1)."""
        if rkey is None:
            rkey = self.rkey
            self.newkey()
        return random.normal(rkey, shape=(self.nr_particles, self.d)) - 5

    def kernel_step(self, kernel_params, x, stepsize):
        """Update kernel_params <- kernel_params + stepsize * \gradient_{kernel_params} KSD_{kernel_params}(q, p)^2
        Arguments:
        * kernel_params: scalar or np.array of shape (d,)
        * x: np.array of shape (n,d).
        * stepsize: scalar
        * log: dict for logging stuff"""
        def ksd_sq(kernel_params):
            return stein.ksd_squared(x, self.logp, self.kernel(kernel_params))
        gradient = grad(ksd_sq)(kernel_params)

#        # normalize gradient
#        gradient = gradient / np.linalg.norm(gradient)

#        # clip gradient
#        gradient = optimizers.clip_grads(gradient, 1)

        return kernel_params + stepsize * gradient, gradient # TODO: (modularity) this should be done by optimizer object
    kernel_step = jit(kernel_step, static_argnums=0)

    def svgd_step(self, x, kernel_params, stepsize):
        """Update X <- X + \phi^*_{p,q}(X)

        Arguments:
        * x: np.array of shape (n,d).
        * h: scalar or np.array of shape (d,)
        * stepsize: scalar"""
        return x + stepsize * phistar(x, self.logp, self.kernel(kernel_params))
    svgd_step = jit(svgd_step, static_argnums=0)

    def train(self, rkey, kernel_params, lr, svgd_stepsize, n_steps):
        kernel_params = np.array(kernel_params, dtype=np.float32) # cast to float so it works with grad

        x = self.initialize(rkey)
        log = metrics.initialize_log(self)
        log["x0"] = x
        log["ksd_gradients"] = []
        if self.snapshot_iter is not None:
            log["particle_snapshots"] = []
            log["kernel_params_snapshots"] = []
        ksd_pre = []
        ksd_post = []
        for i in tqdm(range(n_steps)):
            log = metrics.update_log(self, i, x, log, kernel_params)
            ksd_pre.append(stein.ksd_squared(x, self.logp, kernel_params))

            # update kernel_params
            kernel_params, gradient = self.kernel_step(kernel_params, x, lr)
            ksd_post.append(stein.ksd_squared(x, self.logp, kernel_params))
            log["ksd_gradients"].append(gradient)
            if np.any(np.isnan(kernel_params)):
                log["interrupt_iter"] = i
                log["last_kernel_params"] = log["desc"]["kernel_params"][i-1]
                warnings.warn(f"NaNs detected in kernel_params at iteration {i}. Training interrupted.", RuntimeWarning)
                break

            # update x
            x = self.svgd_step(x, kernel_params, svgd_stepsize)
            if np.any(np.isnan(x)):
                log["interrupt_iter"] = i
                warnings.warn(f"NaNs detected in x at iteration {i}. Kernel_Params is fine, which means NaNs come from update. Training interrupted.", RuntimeWarning)
                break

            if i in self.snapshot_iter:
                log["particle_snapshots"].append(x)
                log["kernel_params_snapshots"].append(np.exp(kernel_params))

        log["metric_names"] = self.dist.metric_names
        log["ksd_pre"] = np.array(ksd_pre)
        log["ksd_post"] = np.array(ksd_post)
        log["ksd_gradients"] = np.array(log["ksd_gradients"])

        return x, log






    @partial(verbose_jit, static_argnums=(0, 4))
    def svgd(self, x0, stepsize, bandwidth, adaptive_kernel=False):
        """
        IN:
        * rkey: random seed
        * stepsize is a float
        * bandwidth is an np array of length d: bandwidth parameter for RBF kernel.
          The bandwidth is fixed throughout training.
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
            if adaptive_kernel:
                _bandwidth = kernels.median_heuristic(x)
            else:
                _bandwidth = bandwidth

            log = metrics.update_log(self, i, x, log, _bandwidth)
            x = update(x, self.logp, stepsize, self.kernel(_bandwidth), None, adagrad, historical_grad)

            rkey = random.split(rkey)[0]
            x = x + random.normal(rkey, shape=x.shape) * self.noise

            return [x, log, historical_grad, rkey]

        x, log, *_ = lax.fori_loop(0, self.n_iter, update_fun, [x, log, historical_grad, rkey])
        log["metric_names"] = self.dist.metric_names
        return x, log
