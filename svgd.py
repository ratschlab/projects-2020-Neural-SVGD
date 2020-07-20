import jax.numpy as np
from jax import jit, vmap, random, value_and_grad

import time
from tqdm import tqdm

from functools import partial
import json_tricks as json

import utils
from utils import NanError
import metrics
import stein


class Optimizer():
    def __init__(self, opt_init, opt_update, get_params):
        """opt_init, opt_update, get_params are the three functions obtained
        from a stax.optimizer call."""
        self.init = jit(opt_init)
        self.update = jit(opt_update)
        self.get_params = jit(get_params)


@partial(jit, static_argnums=1)
def init_svgd(key, particle_shape):
    return random.normal(key, particle_shape) * 2 - 6


class SVGD():
    def __init__(self, target, n_particles, optimizer_svgd, kernel):
        """
        target: instance of class metrics.Distribution

        kernel needs to have pure methods
           kernel.init(rkey, x, y) -> params
           kernel.apply(params, x, y) -> scalar

        optimizer_svgd needs to have pure methods
           optimizer_svgd.init(params) -> state
           optimizer_svgd.update(rkey, gradient, state) -> state
           optimizer_svgd.get_params(state) -> params
        """
        self.target = target
        self.n_particles = n_particles
        self.kernel = kernel
        self.opt = optimizer_svgd

        self.n = self.n_particles
        self.particle_shape = (self.n, self.target.d)

#    @partial(jit, static_argnums=0)
    def phistar(self, particles, kernel_params):
        kernel_fn = lambda x, y: self.kernel.apply(kernel_params, x, y)
        return stein.phistar(particles, self.target.logpdf, kernel_fn)

    def ksd_squared(self, kernel_params, particles):
        kernel_fn = lambda x, y: self.kernel.apply(kernel_params, x, y)
        return stein.ksd_squared(particles, self.target.logpdf, kernel_fn)

#    @partial(jit, static_argnums=0)
    def negative_ksd_squared_batched(self, kernel_params, particles):
        """Mean - KSD^2 for a batch of particles."""
        if particles.ndim < 3:
            particles = np.expand_dims(particles, 0)  # batch dimension
        return - np.mean(vmap(self.ksd_squared, (None, 0))(kernel_params, particles))

    def train_kernel(self, key, n_iter, ksd_steps, svgd_steps, opt_ksd):
        """Train the kernel parameters
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
        def current_step(i, j, steps):
            return i*steps + j
        key1, _ = random.split(key)

        particles = init_svgd(key, self.particle_shape)
        opt_svgd_state = self.opt.init(particles)

        x_dummy = self.target.sample(1)
        x_dummy = np.reshape(x_dummy, newshape=(self.target.d,))
#        x_dummy = np.squeeze(x_dummy)
#        x_dummy = np.stack([x_dummy, x_dummy])

        kernel_params = self.kernel.init(key1, x_dummy, x_dummy)
        opt_ksd_state = opt_ksd.init(kernel_params)

        log = dict()
        def train():
            nonlocal opt_svgd_state
            nonlocal opt_ksd_state
            nonlocal log
            for i in tqdm(range(n_iter)):
                # update particles:
                kernel_params = opt_ksd.get_params(opt_ksd_state)
                particle_batch = []
                for j in range(svgd_steps):
                    step = current_step(i, j, svgd_steps)

                    particles = self.opt.get_params(opt_svgd_state)
                    gp = -self.phistar(particles, kernel_params)
                    opt_svgd_state = self.opt.update(step, gp, opt_svgd_state)

                    particle_batch.append(particles)
                    metrics.append_to_log(log, self.target.compute_metrics(particles))
                    metrics.append_to_log(log, {"mean": np.mean(particles, axis=0),
                                                "var": np.var(particles, axis=0)})
                    if utils.is_nan(gp):
                        raise NanError("Particle update is NaN. Interrupting training.")

                # update kernel_params:
                particle_batch = np.asarray(particle_batch, dtype=np.float32)
                inner_updates = []
                ksds = []
                for j in range(ksd_steps):
                    step = current_step(i, j, ksd_steps)
                    kernel_params = opt_ksd.get_params(opt_ksd_state)
                    ksd, gk = value_and_grad(self.negative_ksd_squared_batched)(kernel_params, particle_batch)
                    ksd = -ksd
                    opt_ksd_state = opt_ksd.update(step, gk, opt_ksd_state)

                    ksds.append(ksd)
                    utils.warn_if_nan(ksd)
                    if utils.is_nan(gk):
                        raise NanError("KSD update gradient is NaN. Interrupting training.")

                update_log = {
                    "ksd": ksds
                }
                metrics.append_to_log(log, update_log)

            log["particles"] = particles
            return None

        try:
            train()
            log["Interrupted because of NaN"] = False
        except NanError as e:
            print(e)
            log["Interrupted because of NaN"] = True

        log = utils.dict_asarray(log)
        return kernel_params, log

    # @partial(jit, static_argnums=(0, 3))
    def sample(self, kernel_params, key, n_iter):
        """Sample from the (approximation of the) target distribution using the current kernel parameters."""
        particles = init_svgd(key, self.particle_shape)
        opt_svgd_state = self.opt.init(particles)

        log = dict()
        for i in tqdm(range(n_iter)):
            particles = self.opt.get_params(opt_svgd_state)
            gp = -self.phistar(particles, kernel_params)
            opt_svgd_state = self.opt.update(i, gp, opt_svgd_state)
            metrics.append_to_log(log, {"mean": np.mean(particles, axis=0),
                                        "var": np.var(particles, axis=0)})

        return self.opt.get_params(opt_svgd_state), log
