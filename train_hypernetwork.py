import jax.numpy as np
from jax import grad, jit, vmap, random, value_and_grad
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

class Optimizer():
    def __init__(self, opt_init, opt_update, get_params):
        """opt_init, opt_update, get_params are the three functions optained from a stax.optimizer call."""
        self.init = jit(opt_init)
        self.update = jit(opt_update)
        self.get_params = jit(get_params)

@partial(jit, static_argnums=1)
def init_svgd(key, particle_shape):
    return random.normal(key, particle_shape) * 2 - 3

class SVGD():
    def __init__(self, target, n_particles, optimizer_svgd, kernel, hypernetwork):
        self.target = target
        self.n_particles = n_particles
        self.kernel = kernel
        self.opt = optimizer_svgd
        self.hypernetwork = hypernetwork

        self.n = self.n_particles
        self.particle_shape = (self.n, self.target.d)
        self.kernel_param_shape = () # 1d for now, TODO

    def _phistar(self, particles, kernel_params):
        return stein.phistar(particles, self.target.logpdf, self.kernel(kernel_params))

#    @partial(jit, static_argnums=0)
    def phistar(self, particles, params):
        inshape = particles.shape
        if particles.ndim < 3:
            particles = np.expand_dims(particles, 0) # batch dimension NOTE: we never apply phistar to an actual batch. But hypernetwork always needs a batch dimension in the input. So shape needs to be (1, n, d)
        kernel_params = self.hypernetwork.apply(params, particles)
        return np.reshape(vmap(self._phistar)(particles, kernel_params), newshape=inshape)

    def _ksd_squared(self, kernel_params, particles):
        return stein.ksd_squared(particles, self.target.logpdf, self.kernel(kernel_params))

#    @partial(jit, static_argnums=0)
    def ksd_squared(self, params, particles):
        """Mean KSD^2 for a batch of particles."""
        if particles.ndim < 3:
            particles = np.expand_dims(particles, 0) # batch dimension
        kernel_params = self.hypernetwork.apply(params, particles)
        return np.mean(vmap(self._ksd_squared)(kernel_params, particles))

    def train_kernel(self, key, n_iter, ksd_steps, svgd_steps, opt_ksd):
        """Train the hypernetwork to output the correct kernel parameters.
        Training goes over one SVGD run.

        Arguments
        ---------
        key : jax.random.PRNGKey
        n_iter : nr of iterations in total
        ksd_steps : nr of ksd steps per iteration
        svgd_steps : nr of svgd steps per iteration
        opt_ksd : instance of class Optimizer.

        Returns
        -------
        Updated kernel parameters.
        """
        hypernetwork = self.hypernetwork
        def current_step(i, j, steps):
            return i*steps + j
        key1, key2 = random.split(key)

        particles = init_svgd(key1, self.particle_shape)
        opt_svgd_state = self.opt.init(particles)

        particles = np.expand_dims(particles, 0) # batch dimension
        params = hypernetwork.init(key2, particles)
        opt_ksd_state = opt_ksd.init(params)

        log = dict()
        for i in range(n_iter):
            # update particles:
            params = opt_ksd.get_params(opt_ksd_state)
            particle_batch = []
            for j in range(svgd_steps):
                step = current_step(i, j, svgd_steps)

                particles = self.opt.get_params(opt_svgd_state)
                gp = -self.phistar(particles, params) # TODO gradient has wrong extra batch dim
                opt_svgd_state = self.opt.update(step, gp, opt_svgd_state)

                particle_batch.append(particles)
                utils.warn_if_nan(gp)

            # update network params:
            particle_batch = np.asarray(particle_batch, dtype=np.float32)
            log = metrics.append_to_log(log, {"particles": particle_batch})
            inner_updates = []
            ksds = []
            gradients = []
            for j in range(ksd_steps):
                step = current_step(i, j, ksd_steps)
                params = opt_ksd.get_params(opt_ksd_state)
                ksd, gk = value_and_grad(self.ksd_squared)(params, particle_batch)
                opt_ksd_state = opt_ksd.update(step, gk, opt_ksd_state)

#                inner_updates.append(params)
                ksds.append(ksd)
                gradients.append(gk)
                utils.warn_if_nan(ksd)
                utils.warn_if_nan(gk)
            update_log = {
                "ksd": ksds,
                "gradients": gradients,
            }
            log = metrics.append_to_log(log, update_log)

        return params, log

    # @partial(jit, static_argnums=(0, 3))
    def sample(self, key, params, n_svgd_steps):
        """Sample from the (approximation of the) target distribution using the current hypernetwork parameters."""
        particles = init_svgd(key, self.particle_shape)
        opt_svgd_state = self.opt.init(particles)

        for i in range(n_svgd_steps):
            particles = self.opt.get_params(opt_svgd_state)
            kernel_params = hypernetwork.apply(params, particles)
            gp = -self.phistar(particles, kernel_params)
            opt_svgd_state = self.opt.update(i, gp, opt_svgd_state)

        return self.opt.get_params(opt_svgd_state)


if __name__ == "main":
    # config
    kernel = kernels.ard
    target = metrics.Gaussian(0, 10) # target dist
    n = 500 # nr particles

    lr_svgd = 1
    lr_ksd = 1e-2

    n_svgd_steps = 300
    ksd_steps_per_svgd_step = 1

    opt_svgd = train.Optimizer(*optimizers.sgd(step_size=lr_svgd))
    opt_ksd  = train.Optimizer(*optimizers.momentum(step_size=lr_ksd, mass=0.9))

    today = time.strftime("%Y-%m-%d")
    logdir = f"results/{today}"
    logfile = f"results/{today}/"+config["experiment_name"]

    try:
        os.makedirs(logdir)
    except:
        pass

    # run SVGD
    print("Learning kernel parameters...")
    s = train.SVGD(target, n, kernel, lr_svgd, opt_svgd)
    params, log = s.train_kernel(rkey, n_svgd_steps, ksd_steps_per_svgd_step, lr_ksd, opt_ksd)

    # write to logs
    print(f"Writing results to {logfile}.")
    with open(logfile, "w") as f:
        f.write()
