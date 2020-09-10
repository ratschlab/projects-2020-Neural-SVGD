import jax.numpy as np
from jax import jit, vmap, random, value_and_grad
from jax.experimental import optimizers
from jax.ops import index_update, index
import haiku as hk
import jax
import numpy as onp

import traceback
import time
from tqdm import tqdm
from functools import partial
import json_tricks as json

import utils
import metrics
import stein
import kernels
import nets
import distributions
import kernel_learning

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


class Particles():
    """
    Container class for particles, particle optimizer,
    particle update step method, and particle metrics.
    """
    def __init__(self,
                 key,
                 gradient: callable,
                 proposal,
                 target=None,
                 n_particles: int = 400,
                 learning_rate=0.1):
        """
        Arguments
        ----------
        gradient: takes in args (particles, params) and returns
            an array of shape (n, d), interpreted as grad(loss)(particles).
        proposal: instances of class distributions.Distribution
        optimizer needs to have pure methods
           optimizer.init(params) -> state
           optimizer.update(key, gradient, state) -> state
           optimizer.get_params(state) -> params
        """
        self.gradient = gradient
        self.target = target
        self.proposal = proposal
        self.n_particles = n_particles

        # optimizer for particle updates
        self.opt = Optimizer(*optimizers.sgd(learning_rate))
        self.threadkey, subkey = random.split(key)
        self.initialize_optimizer(subkey)
        self.step_counter = 0
        self.rundata = {}
#         self.initialize_groups()

    def initialize_optimizer(self, key):
        particles = self.init_particles(key)
        self.optimizer_state = self.opt.init(particles)
        return None

    def init_particles(self, key):
        particles = self.proposal.sample(self.n_particles, key=key)
        return particles

    def get_params(self):
        return self.opt.get_params(self.optimizer_state)

    @partial(jit, static_argnums=0)
    def _step(self, optimizer_state, params, step_counter):
        """
        Updates particles in direction of the gradient.

        params can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or nothing.
        """
        particles = self.opt.get_params(optimizer_state)
        gradient, grad_aux = self.gradient(particles, params, aux=True)
        optimizer_state = self.opt.update(step_counter, gradient, optimizer_state)
        auxdata = gradient
        return optimizer_state, auxdata

    def step(self, params):
        """Log rundata, take step, update loglikelihood. Mutates state"""
        updated_optimizer_state, auxdata = self._step(self.optimizer_state, params, self.step_counter)
        self.log(auxdata)
        self.optimizer_state = updated_optimizer_state # take step
        self.step_counter += 1
        return None

    def log(self, auxdata):
        gradient = auxdata
        particles = self.get_params()
        mean_auxdata = np.mean(auxdata, axis=0)
        metrics.append_to_log(self.rundata, {
            "step": self.step_counter,
            "gradient_norm": np.linalg.norm(gradient),
            "mean": np.mean(particles, axis=0),
            "std": np.std(particles, axis=0),
        })


class SDLearner():
    """Parametrize function to learn the stein discrepancy"""
    def __init__(self,
                 key,
                 target,
                 sizes: list,
                 learning_rate: float = 0.01,
                 lambda_reg: float = 1,
                 std_normalize=None):
        """
        When sizes = [d] and no biases, then this is equivalent to just maximizing the
        vanilla RBF parameters (bandwidth / ARD covariance matrix).
        """
        self.sizes = sizes
        self.target = target
        self.lambda_reg = lambda_reg
        self.std_normalize = std_normalize
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        # small experiments suggest it's better to have no skip connection
        # and to not use an activation on the final layer
        self.mlp = nets.build_mlp(self.sizes, name="MLP", skip_connection=False,
                                  with_bias=True, activate_final=False)
        self.opt = kernel_learning.Optimizer(*optimizers.adam(learning_rate))
        self.step_counter = 0
        self.initialize_optimizer(subkey)
        self.rundata = {}

    def initialize_optimizer(self, key=None, keep_params=False):
        """Initialize optimizer. If keep_params=True, then only the learning
        rate schedule is reinitialized, otherwise the model parameters are
        also reinitialized."""
        if keep_params:
            self.optimizer_state = self.opt.init(self.get_params())
        else:
            x_dummy = np.ones(self.target.d)
            if key is None:
                self.threadkey, subkey = random.split(self.threadkey)
            else:
                key, subkey = random.split(key)
            init_params = self.mlp.init(subkey, x_dummy)
            self.optimizer_state = self.opt.init(init_params)
        return None

    def get_params(self):
        return self.opt.get_params(self.optimizer_state)

    @partial(jit, static_argnums=0)
    def loss_fn(self, params, samples):
        f = self.get_f(params)
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            samples, self.target.logpdf, f, aux=True)
        def fnorm(x): return np.linalg.norm(f(x))**2
        regularizer_term = np.mean(vmap(fnorm)(samples))
        aux = [stein_discrepancy, regularizer_term, stein_aux]
        return -stein_discrepancy + self.lambda_reg * regularizer_term, aux

    def get_f(self, params=None):
        """return f(\cdot)"""
        if params is None:
            params = self.get_params()
        def f(x): return self.mlp.apply(params, None, x)
        return f

    @partial(jit, static_argnums=0)
    def _step(self, optimizer_state, samples, step: int):
        # update step
        params = self.opt.get_params(optimizer_state)
        [loss, aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        aux.append(loss)
        return optimizer_state, aux

    def step(self, samples):
        """Step and mutate state"""
        self.optimizer_state, aux = self._step(
            self.optimizer_state, samples, self.step_counter)
        self.log(aux)
        self.step_counter += 1
        return None

    def log(self, aux):
        sd, fnorm, stein_aux, loss = aux
        drift, repulsion = stein_aux # shape (2, d)
        params = self.get_params()
        metrics.append_to_log(self.rundata, {
            "step": self.step_counter,
            "training_sd": sd,
            "loss": loss,
            "fnorm": fnorm,
            "mean_drift": np.mean(drift),
            "mean_repulsion": np.mean(repulsion),
        })
        return

    def validate(self, validation_samples):
        params = self.get_params()
        val_loss, auxdata = self.loss_fn(params, samples)
        val_sd, fnorm, stein_aux, loss = auxdata
        metrics.append_to_log(self.rundata, {
            "validation_loss": (self.step_counter, val_loss),
            "validation_sd": (self.step_counter, val_sd),
        })
        return

    def train(self, samples, key=None, n_steps=100, proposal=None, batch_size=200, noise_level=0):
        for _ in tqdm(range(n_steps), disable=disable_tqdm):
            if proposal is not None:
                samples = proposal.sample(batch_size)
            if key is None:
                self.threadkey, key = random.split(self.threadkey)
            try:
                key, subkey = random.split(key)
                step_samples = samples + random.normal(subkey, samples.shape)*noise_level
                self.step(step_samples)
            except KeyboardInterrupt:
                return
        return samples


class KernelLearner():
    """Parametrize kernel to maximize the kernelized Stein discrepancy."""
    def __init__(self,
                 key,
                 target,
                 sizes: list,
                 activation_kernel: callable,
                 learning_rate: float = 0.01,
                 lambda_reg: float = 1,
                 scaling_parameter = False,
                 std_normalize=False):
        """
        When sizes = [d] and no biases, then this is equivalent to just maximizing the
        vanilla RBF parameters (bandwidth / ARD covariance matrix).
        """
        self.sizes = sizes
        self.activation_kernel = activation_kernel
        self.target = target
        self.lambda_reg = lambda_reg
        self.scaling_parameter = scaling_parameter
        self.std_normalize = std_normalize
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        self.mlp = nets.build_mlp(self.sizes, name="MLP", skip_connection=True,
                                  with_bias=False)
        self.decoder = nets.build_mlp(self.sizes, name="decoder",
                                      skip_connection=False, with_bias=False)
        self.opt = Optimizer(*optimizers.adam(learning_rate))
        self.step_counter = 0
        self.initialize_optimizer(subkey)
        self.rundata = {}

    def initialize_optimizer(self, key=None, keep_params=False):
        """Initialize optimizer. If keep_params=True, then only the learning
        rate schedule is reinitialized, otherwise the model parameters are
        also reinitialized."""
        if keep_params:
            self.optimizer_state = self.opt.init(self.get_params())
        else:
            x_dummy = np.ones(self.target.d)
            if key is None:
                self.threadkey, subkey = random.split(self.threadkey)
            else:
                key, subkey = random.split(key)
            init_params_enc = self.mlp.init(subkey, x_dummy)
            init_params_dec = self.decoder.init(
                subkey, self.mlp.apply(init_params_enc, None, x_dummy))
            if self.scaling_parameter:
                init_normalizing_constant = 1.
                init_params = (init_params_enc, init_params_dec, init_normalizing_constant)
            else:
                init_params = (init_params_enc, init_params_dec)
            self.optimizer_state = self.opt.init(init_params)
        return None

    def get_params(self):
        return self.opt.get_params(self.optimizer_state)

    def get_kernel(self, params=None):
        if params is None:
            params = self.get_params()
        if self.scaling_parameter:
            enc_params, _, norm = params
        else:
            enc_params, *_ = params
        def kernel(x, y):
            x, y = np.asarray(x), np.asarray(y)
            k = self.activation_kernel(self.mlp.apply(enc_params, None, x),
                                       self.mlp.apply(enc_params, None, y))
            return k*norm if self.scaling_parameter else k
        return kernel

    def loss_fn(self, params, samples):
        kernel = self.get_kernel(params)
        ksd_squared, std = stein.ksd_squared_u(
            samples, self.target.logpdf, kernel, include_stddev=True)
        phistar = self.get_phistar(samples, params=params)
        def phi_norm(x): return np.linalg.norm(phistar(x))**2
        regularizer_term = np.mean(vmap(phi_norm)(samples))
        ksd = np.sqrt(np.clip(ksd_squared, a_min=1e-6))
        aux = [ksd, ksd_squared, std, regularizer_term]
        if self.std_normalize:
            return -ksd_squared/std + self.lambda_reg * regularizer_term, aux
        else:
            return -ksd_squared +     self.lambda_reg * regularizer_term, aux

    def get_phistar(self, samples, params=None):
        """return phistar(\cdot)"""
        if params is None:
            params = self.get_params()
        kernel = self.get_kernel(params)
        def phistar(x):
            return stein.phistar_i(x, samples, self.target.logpdf, kernel, aux=False)
        return phistar

    @partial(jit, static_argnums=0)
    def _step(self, optimizer_state, samples, step: int):
        # update step
        params = self.opt.get_params(optimizer_state)
        [loss, aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        aux.append(loss)
        aux.append(g)
        aux.append(params) # params before update
        return optimizer_state, aux

    def step(self, samples):
        """Step and mutate state"""
        self.optimizer_state, aux = self._step(
            self.optimizer_state, samples, self.step_counter)
        self.log(aux)
        self.step_counter += 1
        return None

    def log(self, aux):
        ksd, ksd_squared, std, reg, full_loss, grad, params_pre = aux
        params = self.get_params()
        metrics.append_to_log(self.rundata, {
            "training_ksd": ksd,
            "bandwidth": 1 / np.squeeze(params[0]["MLP/~/linear_0"]["w"])**2,
            "std_of_ksd": std,
            "ksd_squared": ksd_squared,
            "loss": full_loss,
            "regularizer": reg,
            "update-to-weight-ratio": utils.compute_update_to_weight_ratio(
                params_pre[0], params[0])
        })
        if self.scaling_parameter:
            metrics.append_to_log(self.rundata, {
                "normalizing_const": params[2],
            })

    def train(self, samples, n_steps=100, proposal=None, batch_size=200):
        for _ in tqdm(range(n_steps), disable=disable_tqdm):
            if proposal is not None:
                samples = proposal.sample(batch_size)
            try:
                self.step(samples)
            except KeyboardInterrupt:
                return
        return
