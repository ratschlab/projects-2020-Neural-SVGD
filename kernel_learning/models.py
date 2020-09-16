import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, tree_util
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

import os
on_cluster = not os.getenv("HOME") == "/home/lauro"
disable_tqdm = on_cluster

from collections import Mapping

class Logger():
    def __init__(self):
        self.data = {}

    def write(data: Mapping[str, np.ndarray], reducer: callable=None):
        if reducer is not None:
            data = {k: reducer(v) for k, v in data.items()}
        metrics.append_to_log(self.data, data)
        return

    def reset(self):
        self.data = {}


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
                 learning_rate=0.1,
                 only_training=False):
        """
        Arguments
        ----------
        gradient: takes in args (params, key, particles) and returns
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
        self.initialize_groups(only_training=only_training)

    def initialize_groups(self, key=None, only_training=False):
        """Split particles into groups: training and validation"""
        self.group_names = ("training", "validation") # TODO make this a dict or namedtuple
        if only_training:
            self.group_idx = np.arange(1, self.n_particles), [0]
        else:
            if key is None:
                self.threadkey, key = random.split(self.threadkey)
            key, subkey = random.split(key)
            idx = random.permutation(subkey, np.arange(self.n_particles))
            self.group_idx = idx.split(2)
        return None

    def initialize_optimizer(self, key=None, particles=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        if particles is None:
            particles = self.init_particles(key)
        self.optimizer_state = self.opt.init(particles)
        return None

    def init_particles(self, key):
        particles = self.proposal.sample(self.n_particles, key=key)
        return particles

    def get_params(self, split_by_group=False):
        if split_by_group:
            return [self.opt.get_params(self.optimizer_state)[idx]
                    for idx in self.group_idx]
        else:
            return self.opt.get_params(self.optimizer_state)

    def perturb(self, noise_level=1e-2, key=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        particles = self.get_params()
        particles += random.normal(key, shape=particles.shape)
        self.initialize_optimizer(particles=particles)
        return None

    @partial(jit, static_argnums=0)
    def _step(self, key, optimizer_state, params, step_counter):
        """
        Updates particles in direction of the gradient.

        params can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or nothing.
        """
        particles = self.opt.get_params(optimizer_state)
        gradient, grad_aux = self.gradient(params, key, particles, aux=True)
        optimizer_state = self.opt.update(step_counter, gradient, optimizer_state)
        auxdata = gradient
        return optimizer_state, auxdata, grad_aux # grad_aux is a dict

    def step(self, params, key=None):
        """Log rundata, take step, update loglikelihood. Mutates state"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        updated_optimizer_state, auxdata, grad_aux = self._step(
            key, self.optimizer_state, params, self.step_counter)
        self.log(auxdata, grad_aux)
        self.optimizer_state = updated_optimizer_state # take step
        self.step_counter += 1
        return None

    def log(self, auxdata, grad_aux):
        gradient = auxdata
        particles = self.get_params()
        metrics.append_to_log(self.rundata, grad_aux)
        metrics.append_to_log(self.rundata, {
            "step": self.step_counter,
            "gradient_norm": np.linalg.norm(gradient),
            "mean": np.mean(particles, axis=0),
            "std": np.std(particles, axis=0),
        })
        for k, idx in zip(self.group_names, self.group_idx): # TODO: iterate thru particle groups directly instead
            metrics.append_to_log(self.rundata, {
                f"{k}_mean": np.mean(particles[idx], axis=0),
                f"{k}_std":  np.std(particles[idx], axis=0),
            })
        return


class SDLearner():
    """Parametrize function to maximize the stein discrepancy"""
    def __init__(self,
                 key,
                 target,
                 sizes: list,
                 learning_rate: float = 0.01,
                 lambda_reg: float = 1,
                 std_normalize=None):
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
        self.opt = Optimizer(*optimizers.adam(learning_rate))
        self.step_counter = 0
        self.initialize_optimizer(subkey)
        self.rundata = {}
        self.frozen_states = []

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

    def kl_gradient(self, params, _, particles, aux=False):
        """Compute gradient vector field based on params and particles."""
        f = self.get_f(params)
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            particles, self.target.logpdf, f, aux=True)
        def fnorm(x): return np.linalg.norm(f(x))**2
        regularizer_term = np.mean(vmap(fnorm)(particles))
        auxdata = {"sd": stein_discrepancy,
                   "fnorm": regularizer_term}
        if aux:
            return - f(particles), auxdata
        else:
            return - f(particles)

    def _step_unjitted(self, optimizer_state, samples, step: int):
        # update step
        params = self.opt.get_params(optimizer_state)
        [loss, aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        aux.append(loss)
        return optimizer_state, aux
    _step = jit(_step_unjitted, static_argnums=0)

    def step(self, samples, disable_jit=False):
        """Step and mutate state"""
        step_fn = self._step_unjitted if disable_jit else self._step
        updated_optimizer_state, aux = step_fn(
            self.optimizer_state, samples, self.step_counter)
        if any([np.any(np.isnan(leaf))
                for leaf in tree_util.tree_leaves(updated_optimizer_state)]):
            raise FloatingPointError("NaN detected!")
        self.optimizer_state = updated_optimizer_state
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
        val_loss, auxdata = self.loss_fn(params, validation_samples)
        val_sd, fnorm, stein_aux = auxdata
        metrics.append_to_log(self.rundata, {
            "validation_loss": (self.step_counter, val_loss),
            "validation_sd": (self.step_counter, val_sd),
        })
        return

    def train(self, samples, validation_samples, key=None, n_steps=100, noise_level=0, catch_nan_errors=False):
        """
        Arguments:
        * samples: batch to train on
        * proposal: distribution batch is sampled from. Used to validate (or to
        sample when sample_every is True)
        * batch_size: nr of samples (use when not passing samples)
        """
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def step(key, samples):
            step_samples = samples + random.normal(key, samples.shape)*noise_level
            self.step(step_samples)
            if self.step_counter % 20 == 0:
                self.validate(validation_samples)

        for _ in range(n_steps):
            try:
                key, subkey = random.split(key)
                step(subkey, samples)
            except FloatingPointError as err:
                if catch_nan_errors:
                    return
                else:
                    raise err from None
        return

    def train_sampling_every_time(self):
        raise NotImplementedError()

    def freeze_state(self):
        """Stores current params, log, and step_counter"""
        self.frozen_states.append((self.step_counter,
                                   self.get_params(),
                                   self.rundata))
        return

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
        ksd_squared = stein.ksd_squared_u(
            samples, self.target.logpdf, kernel, include_stddev=False)
        phistar = self.get_phistar(samples, params=params)
        def phi_norm(x): return np.linalg.norm(phistar(x))**2
        regularizer_term = np.mean(vmap(phi_norm)(samples))
        ksd = np.sqrt(np.clip(ksd_squared, a_min=1e-6))
        aux = [ksd, ksd_squared, regularizer_term]
        if self.std_normalize: # TODO: disabled this for now
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
        ksd, ksd_squared, reg, full_loss, grad, params_pre = aux
        params = self.get_params()
        metrics.append_to_log(self.rundata, {
            "training_ksd": ksd,
            "bandwidth": 1 / np.squeeze(params[0]["MLP/~/linear_0"]["w"])**2,
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

    def kl_gradient(self, params, _, particles, aux=False):
        """
        Compute gradient vector field based on params and particles.
        params is just the kernel params."""
        kernel = self.get_kernel(params)
        ksd_squared = stein.ksd_squared_u(
            particles, self.target.logpdf, kernel, include_stddev=False)
        phistar = self.get_phistar(particles, params=params)
        def phi_norm(x): return np.linalg.norm(phistar(x))**2
        regularizer_term = np.mean(vmap(phi_norm)(particles))
        auxdata = {"ksd": ksd_squared,
                   "phi_norm_l2": regularizer_term}
        if aux:
            return vmap(phistar)(particles), auxdata
        else:
            return vmap(phistar)(particles)

    def train(self, samples, n_steps=100, progress_bar=False):
        for _ in tqdm(range(n_steps), disable=not progress_bar):
            self.step(samples)
        return

    def train_sampling_every_time(self):
        raise NotImplementedError()

class ScoreLearner():
    """Parametrize function to learn the score function grad(log q)
    by maximizing E[grad(f)] under q (equivalent to score matching).

    Gradient of the KL(q \Vert p) is then approximated as f(x) - \log p(x)"""
    def __init__(self,
                 key,
                 target,
                 sizes: list,
                 learning_rate: float = 0.01,
                 lambda_reg: float = 1):
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
        self.opt = Optimizer(*optimizers.adam(learning_rate))
        self.step_counter = 0
        self.initialize_optimizer(subkey)
        self.rundata = {}
        self.frozen_states = []

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
        """
        params: weights of f
        samples: particles sampled from current model q
        """
        score = self.get_score_function(params)
        def score_norm(x): return np.linalg.norm(score(x))**2
        l2_norm = np.mean(vmap(score_norm)(samples))
        def jac_trace(x): return np.trace(jacfwd(score)(x).transpose())
        grad_score = np.mean(vmap(jac_trace)(samples))
        aux = [l2_norm]
        return grad_score + l2_norm/2, aux

    def get_score_function(self, params=None):
        """return \hat s(\cdot)"""
        if params is None:
            params = self.get_params()
        def s(x): return self.mlp.apply(params, None, x)
        return s

    def kl_gradient(self, params, _, particles, aux=False):
        """Compute gradient vector field based on score estimate"""
        score = self.get_score_function(params)
        def kl_gradient_estimate(x):
            """Estimate of \nabla KL = s_q - s_p."""
            return s(x) - grad(target.logpdf)(x)
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            particles, self.target.logpdf, kl_gradient_estimate, aux=True)
        def gradient_norm(x): return np.linalg.norm(f(x))**2
        regularizer_term = np.mean(vmap(gradient_norm)(particles))
        auxdata = {"sd": stein_discrepancy,
                   "gradient_norm_L2": regularizer_term}
        if aux:
            return  vmap(kl_gradient_estimate)(particles), auxdata
        else:
            return  vmap(kl_gradient_estimate)(particles)

    def _step_unjitted(self, optimizer_state, samples, step: int):
        # update step
        params = self.opt.get_params(optimizer_state)
        [loss, aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        aux.append(loss)
        return optimizer_state, aux
    _step = jit(_step_unjitted, static_argnums=0)

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
        val_loss, auxdata = self.loss_fn(params, validation_samples)
        val_sd, fnorm, stein_aux = auxdata
        metrics.append_to_log(self.rundata, {
            "validation_loss": (self.step_counter, val_loss),
            "validation_sd": (self.step_counter, val_sd),
        })
        return

    def train(self, samples, validation_samples, key=None, n_steps=100, noise_level=0, catch_nan_errors=False):
        """
        Arguments:
        * samples: batch to train on
        * proposal: distribution batch is sampled from. Used to validate (or to
        sample when sample_every is True)
        * batch_size: nr of samples (use when not passing samples)
        """
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def step(key, samples):
            step_samples = samples + random.normal(key, samples.shape)*noise_level
            self.step(step_samples)
            if self.step_counter % 20 == 0:
                self.validate(validation_samples)

        for _ in range(n_steps):
            try:
                key, subkey = random.split(key)
                step(subkey, samples)
            except FloatingPointError as err:
                if catch_nan_errors:
                    return
                else:
                    raise err from None
        return

    def train_sampling_every_time(self):
        raise NotImplementedError()

    def freeze_state(self):
        """Stores current params, log, and step_counter"""
        self.frozen_states.append((self.step_counter,
                                   self.get_params(),
                                   self.rundata))
        return
