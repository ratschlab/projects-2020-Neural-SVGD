import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, tree_util, jacfwd, grad
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

from typing import Mapping

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


class Patience():
    """Criterion for early stopping"""
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.impatience = 0
        self.min_validation_loss = None

    def update(self, validation_loss):
        """Returns True when early stopping criterion is met"""
        if self.min_validation_loss is None or self.min_validation_loss > validation_loss:
            self.min_validation_loss = validation_loss
            self.impatience = 0
        else:
            self.impatience += 1
        return

    def out_of_patience(self):
        return self.impatience > self.patience


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


class GradientLearner():
    """
    Superclass. Learn a gradient vector field to transport particles.
    Need to implement the following methods:
    * loss_fn(self, params, samples) --> (loss, aux)
    * _log(self, samples, validation_samples, aux) --> step_log
    * gradient(self, params, key, particles, aux=False) --> particle_gradient
    """
    def __init__(self,
                 target,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 1e-2,
                 lambda_reg: float = 1/2,
                 patience: int = 20):
        """Init."""
        self.sizes = sizes if sizes is not None else [32, 32, target.d]
        self.target = target
        self.lambda_reg = lambda_reg
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        self.mlp = nets.build_mlp(self.sizes, name="MLP", skip_connection=False,
                                  with_bias=True, activate_final=False)
        self.opt = Optimizer(*optimizers.adam(learning_rate))
        self.step_counter = 0
        self.initialize_optimizer(subkey)
        self.rundata = {}
        self.frozen_states = []
        self.patience = Patience(patience)

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
        raise NotImplementedError()

    def gradient(self, params, key, particles, aux=False):
        raise NotImplementedError()

    def _step_unjitted(self, optimizer_state, samples, validation_samples, step: int):
        """update parameters and compute validation loss"""
        params = self.opt.get_params(optimizer_state)
        [loss, loss_aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        _, val_loss_aux = self.loss_fn(params, validation_samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        auxdata = (loss_aux, val_loss_aux)
        return optimizer_state, auxdata
    _step = jit(_step_unjitted, static_argnums=0)

    def step(self, samples, validation_samples, disable_jit=False):
        """Step and mutate state"""
        step_fn = self._step_unjitted if disable_jit else self._step
        updated_optimizer_state, auxdata = step_fn(
            self.optimizer_state, samples, validation_samples, self.step_counter)
#         if any([np.any(np.isnan(leaf))
#                 for leaf in tree_util.tree_leaves(updated_optimizer_state)]):
#             raise FloatingPointError("NaN detected!")
        self.optimizer_state = updated_optimizer_state
        self.write_to_log(self._log(samples, validation_samples, auxdata))
        self.step_counter += 1
        return None

    def log(self, aux): # depends on loss_fn aux
        raise NotImplementedError()

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        metrics.append_to_log(self.rundata, step_data)

    def train(self, samples, validation_samples, key=None, n_steps=100, noise_level=0, catch_nan_errors=False, progress_bar=False):
        """
        Arguments:
        * samples: batch to train on
        * validation_samples
        """
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def step(key, samples):
            step_samples = samples + random.normal(key, samples.shape)*noise_level
            self.step(step_samples, validation_samples)

        for _ in tqdm(range(n_steps), disable=not progress_bar):
            try:
                key, subkey = random.split(key)
                step(subkey, samples)
            except FloatingPointError as err:
                if catch_nan_errors:
                    print("Caught NaN!")
                    return
                else:
                    raise err from None
            val_loss = self.rundata["validation_loss"][-1]
            self.patience.update(val_loss)
            if self.patience.out_of_patience():
                return
        return

    def train_sampling_every_time(self, proposal, key=None, n_steps=100, batch_size=400,
                                  catch_nan_errors=False, progress_bar=True):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        key, subkey = random.split(key)
        validation_samples = proposal.sample(batch_size)
        for _ in tqdm(range(n_steps), disable=not progress_bar):
            try:
                key, subkey = random.split(key)
                samples = proposal.sample(batch_size, key=subkey)
                self.step(samples, validation_samples)
            except FloatingPointError as err:
                if catch_nan_errors:
                    return
                else:
                    raise err from None
        return

    def freeze_state(self):
        """Stores current state as tuple (step_counter, params, rundata)"""
        self.frozen_states.append((self.step_counter,
                                   self.get_params(),
                                   self.rundata))
        return


class SDLearner(GradientLearner):
    """Parametrize function to maximize the stein discrepancy"""
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)

    def get_field(self, params=None):
        """Return vector field v(\cdot) that approximates \nabla KL.
        This is used to compute particle updates x <-- x - eps * v(x)
        v = -f, where f maximizes the stein discrepancy."""
        if params is None:
            params = self.get_params()
        def v(x): return self.mlp.apply(params, None, x)
        return v

    def loss_fn(self, params, samples):
        f = utils.negative(self.get_field(params))
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            samples, self.target.logpdf, f, aux=True)
        l2_f = utils.l2_norm(samples, f)
        loss = -stein_discrepancy + self.lambda_reg * l2_f**2
        aux = [loss, stein_discrepancy, l2_f, stein_aux]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, samples, validation_samples, aux):
        train_aux, val_aux = aux
        loss, sd, l2v, stein_aux = train_aux
        drift, repulsion = stein_aux # shape (2, d)
        val_loss, val_sd, *_ = val_aux
        step_log = {
            "step": self.step_counter,
            "training_loss": loss,
            "training_sd": sd,
            "validation_loss": val_loss,
            "validation_sd": val_sd,
            "l2_norm": l2v,
            "mean_drift": np.mean(drift),
            "mean_repulsion": np.mean(repulsion),
        }
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(params)
        if aux:
            stein_discrepancy, stein_aux = stein.stein_discrepancy(
                particles, self.target.logpdf, v, aux=True)
            auxdata = {"sd": stein_discrepancy}
            return v(particles), auxdata
        else:
            return v(particles)


class KernelLearner(GradientLearner):
    """Parametrize kernel to learn the KSD"""
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.activation_kernel = kernels.get_rbf_kernel(1)

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
            init_params = (init_params_enc, 1.) # (mlp params, scaling_param)
            self.optimizer_state = self.opt.init(init_params)
        return None

    def get_field(self, inducing_samples, params=None):
        """return -phistar(\cdot)"""
        if params is None:
            params = self.get_params()
        kernel = self.get_kernel(params)
        def phistar(x):
            return stein.phistar_i(x, inducing_samples, self.target.logpdf, kernel, aux=False)
        return phistar

    def loss_fn(self, params, samples):
        kernel = self.get_kernel(params)
        ksd_squared = stein.ksd_squared_u(
            samples, self.target.logpdf, kernel, include_stddev=False)
        kxx = np.mean(vmap(kernel)(samples, samples))
        phi = self.get_field(samples, params=params)
        loss = -ksd_squared + self.lambda_reg * utils.l2_norm(samples, phi)**2
        aux = [loss, ksd_squared, kxx]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, samples, validation_samples, aux):
        names = "loss ksd kxx".split()
        step_log = {}
        for group, data in zip(["training", "validation"], aux):
            step_log.update({
                group+"_"+name: value for name, value in zip(names, data)
            })
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(params, inducing_samples=particles)
        if aux:
            return vmap(v)(particles), None
        else:
            return vmap(v)(particles)

    def get_kernel(self, params=None):
        if params is None:
            params = self.get_params()
        mlp_params, scale = params
        def kernel(x, y):
            x, y = np.asarray(x), np.asarray(y)
            k = self.activation_kernel(self.mlp.apply(mlp_params, None, x),
                                       self.mlp.apply(mlp_params, None, y))
            return scale * k
        return kernel


class ScoreLearner(GradientLearner):
    """Neural score matching"""
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        if self.sizes[-1] != self.target.d:
            raise ValueError(f"Output dim must equal target dim; instead "
                             f"received output dim {self.sizes[-1]} and "
                             f"target dim {self.target.d}.")

    def get_field(self, params=None):
        """Return vector field v(\cdot) that approximates
        \nabla KL = grad(log q) - grad(log p).
        This is used to compute particle updates x <-- x - eps * v(x)"""
        score = self.get_score(params)

        def v(x):
            return score(x) - grad(self.target.logpdf)(x) / (2*self.lambda_reg)
        return v

    def get_score(self, params=None):
        """Return callable s approximating score of q = grad(log q)"""
        if params is None:
            params = self.get_params()
        def score(x):
            return self.mlp.apply(params, None, x)
        return score

    def loss_fn(self, params, samples):
        """Evaluates E[div(score)] + l2_norm(score)^2"""
        score = self.get_score(params)
        def divergence(x):
            return np.trace(jacfwd(score)(x))
        mean_divergence = np.mean(vmap(divergence)(samples))
        loss = mean_divergence + self.lambda_reg * utils.l2_norm(samples, score)**2
        aux = [loss, mean_divergence]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, samples, validation_samples, aux):
        train_aux, val_aux = aux
        loss, mean_div = train_aux
        val_loss, val_mean_div = val_aux
        step_log = {
            "step": self.step_counter,
            "training_loss": loss,
            "training_div": mean_div,
            "validation_div": val_mean_div,
            "validation_loss": val_loss,
        }
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(params)
        if aux:
            return v(particles), None
        else:
            return v(particles)
