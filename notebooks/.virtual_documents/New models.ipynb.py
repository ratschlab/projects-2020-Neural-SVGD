get_ipython().run_line_magic("load_ext", " autoreload")

import sys
import copy
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning")
import json
import collections
import itertools
from functools import partial
import importlib

import numpy as onp
from jax.config import config
# config.update("jax_log_compiles", True)
# config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import matplotlib.pyplot as plt

import numpy as onp
import jax
import pandas as pd
import haiku as hk
import ot

import config

import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import kernel_learning

from jax.experimental import optimizers

key = random.PRNGKey(0)

from jax.scipy.stats import norm


class GradientLearner():
    """Learn a gradient vector field to transport particles."""
    def __init__(self,
                 key,
                 target,
                 sizes: list,
                 learning_rate: float = 0.01)
        self.target = target
        self.threadkey, subkey = random.split(key)

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
        raise NotImplementedError()

    def compute_gradient(params, key, particles):
        raise NotImplementedError()

    def _step_unjitted(self, optimizer_state, samples, step: int):
        # update step
        params = self.opt.get_params(optimizer_state)
        [loss, loss_aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        return optimizer_state, loss_aux
    _step = jit(_step_unjitted, static_argnums=0)

    def step(self, samples, disable_jit=False):
        """Step and mutate state"""
        step_fn = self._step_unjitted if disable_jit else self._step
        updated_optimizer_state, aux = step_fn(
            self.optimizer_state, samples, self.step_counter)
        if any([np.any(np.isnan(leaf)) 2 2
                for leaf in tree_util.tree_leaves(updated_optimizer_state)]):
            raise FloatingPointError("NaN detected!")
        self.optimizer_state = updated_optimizer_state
        self.log(aux)
        self.step_counter += 1
        return None

    def log(self, aux): # depends on loss_fn aux
        raise NotImplementedError()

    def validate(self, validation_samples):
        params = self.get_params()
        val_loss, _ = self.loss_fn(params, validation_samples)
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

