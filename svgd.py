import jax.numpy as np
from jax import jit, vmap, random, value_and_grad

import traceback
import time
from tqdm import tqdm
from functools import partial
import json_tricks as json

import utils
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
    def __init__(self, target, n_particles, n_subsamples, optimizer_svgd, kernel):
        """
        Arguments
        ----------
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
        self.n_subsamples = n_subsamples
        self.kernel = kernel
        self.opt = optimizer_svgd

        self.n = self.n_particles
        self.particle_shape = (self.n, self.target.d)

    # can't jit this I think
    def phistar(self, particles, subsample, kernel_params):
        kernel_fn = lambda x, y: self.kernel.apply(kernel_params, x, y)
        return stein.phistar(particles, subsample, self.target.logpdf, kernel_fn)

    def ksd_squared(self, kernel_params, particles_a, particles_b):
        """particles_a and particles_b are two bootstrap samples from the particles."""
        kernel_fn = lambda x, y: self.kernel.apply(kernel_params, x, y)
        return stein.ksd_squared(particles_a, particles_b, self.target.logpdf, kernel_fn)

    @partial(jit, static_argnums=0)
    def negative_ksd_squared_batched(self, kernel_params, particles_a, particles_b):
        """The mean -KSD^2 for a (k, n, d)-sized batch of particles.
        particles_a and particles_b are two bootstrap samples of shape (k, n, d)."""
        if particles_a.ndim < 3 and particles_b.ndim < 3:
            particles_a, particles_b = [np.expand_dims(particles, 0) for particles in (particles_a, particles_b)]  # batch dimension
        return - np.mean(vmap(self.ksd_squared, (None, 0, 0))(kernel_params, particles_a, particles_b))

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

        particles = init_svgd(key, self.particle_shape)
        train_idx, validation_idx = np.arange(self.n_particles).split(2)
        opt_svgd_state = self.opt.init(particles)

        x_dummy = self.target.sample(1)
        x_dummy = np.reshape(x_dummy, newshape=(self.target.d,))

        key = random.split(key)[0]
        kernel_params = self.kernel.init(key, x_dummy, x_dummy)
        opt_ksd_state = opt_ksd.init(kernel_params)

        log = dict(bandwidth = [], ksd_after_kernel_update=[], ksd_after_kernel_update_val=[], mean=[], var=[], ksd_after_svgd_update=[])
        def train():
            nonlocal opt_svgd_state
            nonlocal opt_ksd_state
            nonlocal log
            log["kernel_params"] = []
            rkey = key
            particles = self.opt.get_params(opt_svgd_state)
            particle_batch = [particles]
            for i in tqdm(range(n_iter)):
                # update kernel_params:
                particle_batch = np.asarray(particle_batch, dtype=np.float64) # large batch
                for j in range(ksd_steps):
                    step = current_step(i, j, ksd_steps)
                    kernel_params = opt_ksd.get_params(opt_ksd_state)

                    rkey = random.split(rkey)[0]
                    subsamples = utils.subsample(rkey, particle_batch[:, train_idx, :], self.n_subsamples*2, replace=False, axis=1).split(2, axis=1)
                    ksd_batched, gk = value_and_grad(self.negative_ksd_squared_batched)(kernel_params, *subsamples)
                    opt_ksd_state = opt_ksd.update(step, gk, opt_ksd_state)

                    log["ksd_after_kernel_update"].append(-ksd_batched)
                    rkey = random.split(rkey)[0]
                    validation_subsamples = utils.subsample(rkey, particle_batch[:, validation_idx, :], self.n_subsamples*2, replace=False, axis=1).split(2, axis=1)
                    log["ksd_after_kernel_update_val"].append(-self.negative_ksd_squared_batched(kernel_params, *validation_subsamples))

                    bandwidth = np.exp(kernel_params["ard"]["logh"])**2 # h = sqrt(bandwith)
                    log["bandwidth"].append(bandwidth)
                    log["kernel_params"].append(kernel_params)


                # update particles:
                kernel_params = opt_ksd.get_params(opt_ksd_state)
                particle_batch = []
                ksd_after_kernel_update = []
                for j in range(svgd_steps):
                    step = current_step(i, j, svgd_steps)
                    particles = self.opt.get_params(opt_svgd_state)

                    rkey = random.split(rkey)[0]
                    subsample = utils.subsample(rkey, particles[train_idx], self.n_subsamples, replace=False)

                    gp = -self.phistar(particles, subsample, kernel_params)
                    opt_svgd_state = self.opt.update(step, gp, opt_svgd_state)

                    rkey_a, rkey_b = random.split(rkey)
                    subsample_a = utils.subsample(rkey_a, particles, self.n_subsamples, replace=False, axis=0)
                    subsample_b = utils.subsample(rkey_b, particles, self.n_subsamples, replace=False, axis=0)
                    rkey = rkey_a
                    particle_batch.append(particles)
                    metrics.append_to_log(log, self.target.compute_metrics(particles))
                    metrics.append_to_log(log, {"mean": np.mean(particles, axis=0),
                                                "var": np.var(particles, axis=0),
                                                "ksd_after_svgd_update": self.ksd_squared(kernel_params, subsample_a, subsample_b)})

            log["particles"] = particles
            return None

        try:
            train()
            log["Interrupted because of NaN"] = False
        except FloatingPointError as e:
            print("printing traceback:")
            traceback.print_exc()
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
