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
class SDLearner():
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
        self.mlp = nets.build_mlp(self.sizes, name="MLP", skip_connection=False,
                                  with_bias=False)
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

    def loss_fn(self, params, samples):
        def f(x): return self.mlp.apply(params, None, x)
        stein_discrepancy = stein.stein_discrepancy(
            samples, self.target.logpdf, f, aux=False)
        def fnorm(x): return np.linalg.norm(f(x))**2
        regularizer_term = np.mean(vmap(fnorm)(samples))
        aux = [stein_discrepancy, regularizer_term]
        return -stein_discrepancy + self.lambda_reg * regularizer_term, aux

    def get_f(self):
        """return f(\cdot)"""
        def f(x): return self.mlp.apply(self.get_params(), None, x)
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
        sd, fnorm, loss = aux
        params = self.get_params()
        metrics.append_to_log(self.rundata, {
            "training_sd": sd,
            "loss": loss,
            "fnorm": fnorm,
        })

    def train(self, samples, n_steps=100, proposal=None):
        for _ in tqdm(range(n_steps), disable=disable_tqdm):
            if proposal is not None:
                samples = proposal.sample(400)
            self.step(samples)
        return None
