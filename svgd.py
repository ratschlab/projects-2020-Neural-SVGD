import jax.numpy as np
from jax import jit, vmap, random, value_and_grad
import haiku as hk
import jax

import traceback
import time
from tqdm import tqdm
from functools import partial
import json_tricks as json

import utils
import metrics
import stein

import os
on_cluster = not os.getenv("HOME") == "/home/lauro"
disable_tqdm = on_cluster

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
    def __init__(self, target, n_particles, n_subsamples, optimizer_svgd, kernel, encoder, decoder, subsample_with_replacement: bool, minimize_ksd_variance: bool):
        """
        Arguments
        ----------
        target: instance of class metrics.Distribution

        encoder / decoder need to have pure methods
           encoder.init(key, x, y) -> params
           encoder.apply(params, key, x, y) -> scalar

        optimizer_svgd needs to have pure methods
           optimizer_svgd.init(params) -> state
           optimizer_svgd.update(key, gradient, state) -> state
           optimizer_svgd.get_params(state) -> params
        """
        self.target = target
        self.n_particles = n_particles
        self.n_subsamples = n_subsamples
        self.n_subsamples_for_ksd = n_subsamples*10
        self.subsample_with_replacement = subsample_with_replacement
        self.minimize_ksd_variance = minimize_ksd_variance
        self.encoder = encoder
        self.decoder = decoder
        self.kernel = kernel
        self.opt = optimizer_svgd

        self.n = self.n_particles
        self.particle_shape = (self.n, self.target.d)

        def get_kernel_fn(encoder_params):
            """Returns kernel_fn: x, y --> scalar"""
            def kernel_fn(x, y):
                def enc(x): return self.encoder.apply(encoder_params, None, x)
                return self.kernel(enc(x), enc(y))
            return kernel_fn
        self.get_kernel_fn = get_kernel_fn

    # can't jit this I think
    def phistar(self, particles, subsample, encoder_params):
        """Returns tuple (particle_updates, auxdata)"""
        return stein.phistar(particles, subsample, self.target.logpdf, self.get_kernel_fn(encoder_params))

    def ksd_squared(self, encoder_params, particles_a, particles_b, compute_variance: bool):
        """particles_a and particles_b are two bootstrap samples from the particles. Shape (n, d).
        Returns:
        * estimate of KSD^2
        * estimate of Var(\hat KSD^2), the variance of the linear-time estimator, if compute_variance is True.
        """
        return stein.ksd_squared_l(particles_a, particles_b, self.target.logpdf, self.get_kernel_fn(encoder_params), compute_variance)

    @partial(jit, static_argnums=0)
    def ksd_squared_batched(self, encoder_params, particles_a, particles_b):
        """The mean -KSD^2 for a (k, n, d)-sized batch of particles.
        particles_a and particles_b are two bootstrap samples of shape (k, n, d).
        Returns:
        * estimate of KSD^2
        * estimate of Var(\hat KSD^2), the variance of the linear-time estimator.
        """
        if particles_a.ndim < 3 and particles_b.ndim < 3:
            particles_a, particles_b = [np.expand_dims(particles, 0) for particles in (particles_a, particles_b)]  # batch dimension
        ksds, variances = vmap(self.ksd_squared, (None, 0, 0, None))(encoder_params, particles_a, particles_b, True)
        return np.mean(ksds), np.mean(variances)

    @partial(jit, static_argnums=0)
    def kernel_loss_batched(self, encoder_params, decoder_params, particles_a, particles_b, lambda_reg: float):
        """Regularized KSD
        lam: regularization parameter
        particles have shape (k, n, d)"""
        k, n, d = particles_a.shape
        all_particles = np.concatenate([particles_a, particles_b]).reshape(2*k*n, d)
        def enc(x): return self.encoder.apply(encoder_params, None, x)
        def dec(z): return self.decoder.apply(decoder_params, None, z)
        def autoencoder_loss(x): return np.linalg.norm(x - dec(enc(x)))**2
        autoenc_loss = np.mean(vmap(autoencoder_loss)(all_particles))
        ksd, ksd_var = self.ksd_squared_batched(encoder_params, particles_a, particles_b)
        ksd = np.clip(ksd, a_min=0, a_max=None)
        ksd_var = np.clip(ksd_var, a_min=1e-6, a_max=None)
        aux = [ksd, ksd_var, autoenc_loss]
        if self.minimize_ksd_variance:
            loss = -ksd / ksd_var + lambda_reg * autoenc_loss
        else:
            loss = -ksd + lambda_reg * autoenc_loss
        return loss, aux

    def train_kernel(self, key, n_iter: int, ksd_steps: int, svgd_steps: int, opt_ksd, lambda_reg: float):
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

        key, key1, key2 = random.split(key, 3)
        particles = init_svgd(key1, self.particle_shape)
        opt_svgd_state = self.opt.init(particles)
        leader_idx, train_idx, validation_idx = random.permutation(key2, np.arange(self.n_particles)).split(3)

        x_dummy = self.target.sample(1)
        x_dummy = np.reshape(x_dummy, newshape=(self.target.d,))
        key, key1, key2 = random.split(key, 3)
        encoder_params = self.encoder.init(key1, x_dummy)
        decoder_params = self.decoder.init(key2, self.encoder.apply(encoder_params, None, x_dummy))
        opt_enc_state = opt_ksd.init(encoder_params)
        opt_dec_state = opt_ksd.init(decoder_params)

        rundata = dict()
        def update_kernel(key, opt_enc_state, opt_dec_state, particle_batch, step: int):
            """performs a single ksd update and logs results.
            Returns updated_opt_ksd_state: container for the updated ksd particles and optimizer state
            """
            encoder_params_pre_update = opt_ksd.get_params(opt_enc_state)
            decoder_params_pre_update = opt_ksd.get_params(opt_dec_state)

            # Update
            key, subkey = random.split(key)
            subsamples = utils.subsample(subkey, particle_batch[:, train_idx, :], self.n_subsamples_for_ksd*2, replace=self.subsample_with_replacement, axis=1).split(2, axis=1)
            [regularized_loss, aux], [d_enc, d_dec] = value_and_grad(self.kernel_loss_batched, argnums=(0,1), has_aux=True)(encoder_params_pre_update, decoder_params_pre_update, *subsamples, lambda_reg)
            updated_opt_enc_state = opt_ksd.update(step, d_enc, opt_enc_state)
            updated_opt_dec_state = opt_ksd.update(step, d_dec, opt_dec_state)

            # Compute KSD, validation KSD and log rundata
            ksd_pre_update, autoencoder_loss, ksd_var = aux
            encoder_params = opt_ksd.get_params(updated_opt_enc_state)
            key, subkey = random.split(key)
            validation_subsamples = utils.subsample(subkey, particle_batch[:, validation_idx, :], self.n_subsamples_for_ksd*2, replace=self.subsample_with_replacement, axis=1).split(2, axis=1)
            kernel_fn = self.get_kernel_fn(encoder_params)
            ksd_val, ksd_variance_val = self.ksd_squared_batched(encoder_params_pre_update, *validation_subsamples)
            metrics.append_to_log(rundata, {
                "ksd_before_kernel_update": ksd_pre_update,
                "ksd_before_kernel_update_val": ksd_val,
                "ksd_variance": ksd_var,
                "ksd_variance_val": ksd_variance_val,
                "update_to_weight_ratio": utils.compute_update_to_weight_ratio(encoder_params_pre_update, encoder_params),
                # "sqrt_kxx": vmap(metrics.sqrt_kxx, (None, 0, 0))(kernel_fn, *validation_subsamples), # E[k(x, x)] (has to be finite). O(n^2)
                "regularized_loss": regularized_loss,
                "autoencoder_loss": autoencoder_loss,
            })
            return updated_opt_enc_state, updated_opt_dec_state

        def update_particles(key, opt_svgd_state, encoder_params, step: int):
            """performs a single svgd update and logs results."""
            # Update
            particles = self.opt.get_params(opt_svgd_state)
            key, subkey = random.split(key)
            subsample = utils.subsample(subkey, particles[leader_idx], self.n_subsamples, replace=self.subsample_with_replacement)
            negdKL, auxdata = self.phistar(particles, subsample, encoder_params) # auxdata has shape (n_particles, 2, d)
            dKL = -negdKL
            updated_opt_svgd_state = self.opt.update(step, dKL, opt_svgd_state)

            # compute KSD, validation KSD, and log rundata
            updated_particles = self.opt.get_params(updated_opt_svgd_state)
            key1, key2, key = random.split(key, 3)
            subsamples            = utils.subsample(key1, updated_particles[leader_idx],      self.n_subsamples_for_ksd*2, replace=self.subsample_with_replacement).split(2)
            validation_subsamples = utils.subsample(key2, updated_particles[validation_idx], self.n_subsamples_for_ksd*2, replace=self.subsample_with_replacement).split(2)
            mean_auxdata = np.mean(auxdata, axis=0)
            metrics.append_to_log(rundata, self.target.compute_metrics(updated_particles))
            metrics.append_to_log(rundata, {
                "leader_mean":     np.mean(updated_particles[leader_idx], axis=0),
                "training_mean":   np.mean(updated_particles[train_idx], axis=0),
                "validation_mean": np.mean(updated_particles[validation_idx], axis=0),
                "leader_var":      np.var(updated_particles[leader_idx], axis=0),
                "training_var":    np.var(updated_particles[train_idx], axis=0),
                "validation_var":  np.var(updated_particles[validation_idx], axis=0),
                "ksd_after_svgd_update":     self.ksd_squared(encoder_params, *subsamples, False),
                "ksd_after_svgd_update_val": self.ksd_squared(encoder_params, *validation_subsamples, False),
                "mean_drift": mean_auxdata[0],
                "mean_repulsion": mean_auxdata[1],
            })
            return updated_opt_svgd_state

        def train(key):
            nonlocal opt_svgd_state
            nonlocal opt_enc_state
            nonlocal opt_dec_state
            nonlocal rundata

            encoder_params = opt_ksd.get_params(opt_enc_state)
            particles = self.opt.get_params(opt_svgd_state)
            particle_batch = [particles]
            for i in tqdm(range(n_iter), disable=disable_tqdm):
                particle_batch = np.asarray(particle_batch, dtype=np.float64) # large batch
                for j in range(ksd_steps):
                    step = current_step(i, j, ksd_steps)
                    key, subkey = random.split(key)
                    opt_enc_state, opt_dec_state = update_kernel(subkey, opt_enc_state, opt_dec_state, particle_batch, step)

                particle_batch = []
                kernel_params = opt_ksd.get_params(opt_enc_state)
                for j in range(svgd_steps):
                    step = current_step(i, j, svgd_steps)
                    key, subkey = random.split(key)
                    opt_svgd_state = update_particles(subkey, opt_svgd_state, kernel_params, step)
                    particle_batch.append(self.opt.get_params(opt_svgd_state))
            return None

        try:
            key, subkey = random.split(key)
            train(subkey)
            rundata["Interrupted because of NaN"] = False
        except FloatingPointError as e:
            print("printing traceback:")
            traceback.print_exc()
            rundata["Interrupted because of NaN"] = True

        rundata["update_to_weight_ratio"] = utils.dict_concatenate(rundata["update_to_weight_ratio"])
        rundata["particles"] =     self.opt.get_params(opt_svgd_state)
        rundata["encoder_params"] = opt_ksd.get_params(opt_enc_state)
        rundata["decoder_params"] = opt_ksd.get_params(opt_dec_state)
        rundata.update({
            k: v for k, v in zip(("leader_idx", "train_idx", "validation_idx"), (leader_idx, train_idx, validation_idx))
        })
        rundata = utils.dict_asarray(rundata)
        return opt_ksd.get_params(opt_enc_state), rundata

    # @partial(jit, static_argnums=(0, 3))
    def sample(self, key, encoder_params, n_iter):
        key, subkey = random.split(key)
        particles = init_svgd(subkey, self.particle_shape)
        key, subkey = random.split(key)
        leader_idx, validation_idx = random.permutation(subkey, np.arange(self.n_particles)).split(2)
        opt_svgd_state = self.opt.init(particles)
        rundata = dict()
        for i in tqdm(range(n_iter), disable=disable_tqdm):
            particles = self.opt.get_params(opt_svgd_state)
            key, subkey = random.split(key)
            subsample = utils.subsample(subkey, particles[leader_idx], self.n_subsamples, replace=self.subsample_with_replacement)
            negdKL, auxdata = self.phistar(particles, subsample, encoder_params)
            dKL = -negdKL
            opt_svgd_state = self.opt.update(i, dKL, opt_svgd_state)
            # subsamples for ksd
            key1, key2, key = random.split(key, 3)
            subsamples            = utils.subsample(key1, particles[leader_idx],     self.n_subsamples_for_ksd*2, replace=self.subsample_with_replacement).split(2)
            validation_subsamples = utils.subsample(key2, particles[validation_idx], self.n_subsamples_for_ksd*2, replace=self.subsample_with_replacement).split(2)
            mean_auxdata = np.mean(auxdata, axis=0)
            ksd, ksd_variance =         self.ksd_squared(encoder_params, *subsamples, True)
            ksd_val, ksd_variance_val = self.ksd_squared(encoder_params, *validation_subsamples, True)
            metrics.append_to_log(rundata, {
                "leader_mean": np.mean(particles[leader_idx], axis=0),
                "leader_var":  np.var(particles[leader_idx], axis=0),
                "validation_mean": np.mean(particles[validation_idx], axis=0),
                "validation_var":  np.var(particles[validation_idx], axis=0),
                "ksd_after_svgd_update": ksd,
                "ksd_after_svgd_update_val": ksd_val,
                "ksd_variance": ksd_variance,
                "ksd_variance_val": ksd_variance_val,
                "mean_drift": mean_auxdata[0],
                "mean_repulsion": mean_auxdata[1],
            })
        rundata["particles"] = self.opt.get_params(opt_svgd_state)
        rundata["Interrupted because of NaN"] = False
        return rundata
