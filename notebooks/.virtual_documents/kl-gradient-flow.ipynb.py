import sys
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")

import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, grad
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



class KLFlow():
    def __init__(self,
                 key,
                 target,
                 proposal,
                 n_particles: int = 400,
                 learning_rate=0.1,
                 debugging_config=None):
        """
        Arguments
        ----------
        target, proposal: instances of class distributions.Distribution
        optimizer_svgd needs to have pure methods
           optimizer_svgd.init(params) -> state
           optimizer_svgd.update(key, gradient, state) -> state
           optimizer_svgd.get_params(state) -> params
        """
        self.target = target
        self.proposal = proposal
        self.n_particles = n_particles
        self.debugging_config = debugging_config

        # optimizer for particle updates
        self.opt = kernel_learning.Optimizer(*optimizers.sgd(learning_rate))
        self.step_counter = 0
        self.rundata = {}
        self.threadkey, subkey = random.split(key)
        self.initialize_optimizer(subkey)

        # to track KL
        self.pullback_logp = target.logpdf

    def initialize_optimizer(self, key):
        particles = self.init_particles(key)
        self.optimizer_state = self.opt.init(particles)
        return None

    def init_particles(self, key):
        particle_shape = (self.n_particles, self.target.d)
        particles = self.proposal.sample(self.n_particles, key=key)
        assert particles.shape == particle_shape
        return particles

    def logdiff(self, x, logp):
        return self.proposal.logpdf(x) - logp(x)

    def kl(self, samples, logp):
        return np.mean(vmap(self.logdiff, (0, None))(samples, logp))

    @partial(jit, static_argnums=0)
    def _step(self, optimizer_state, logps, grad_logps, step_counter):
        """
        Updates particles in direction of the SVGD gradient.
        Arguments
            logps: shape (n, d)
            grad_logps: shape (n, d)
        Returns
        * updated optimizer_state
        * dKL: np array of same shape as followers (n, d)
        * auxdata consisting of [mean_drift, mean_repulsion] of shape (n, 2, d)
        """
        particles = self.opt.get_params(optimizer_state)
#         KL, dKL = value_and_grad(self.kl)(particles, logp)
        logqs, grad_logqs = vmap(value_and_grad(proposal.logpdf))(particles)
        KL = np.mean(logqs - logps)
        dKL = np.mean(grad_logqs - grad_logps)
        optimizer_state = self.opt.update(step_counter, dKL, optimizer_state)
        auxdata = (KL, dKL)
        return optimizer_state, auxdata

    def step(self):
        """Log rundata, take step, update loglikelihood. Mutates state"""
        logps, grad_logps = self.logp_value_and_grad()
        updated_optimizer_state, auxdata = self._step(
            self.optimizer_state, logps, grad_logps, self.step_counter)
        self.log(auxdata)
        self.update_logp()
        self.optimizer_state = updated_optimizer_state # take step
        self.step_counter += 1
        return None

    def get_params(self):
        return self.opt.get_params(self.optimizer_state)

    def log(self, auxdata):
        particles = self.opt.get_params(self.optimizer_state)
        KL, dKL = auxdata
        metrics.append_to_log(self.rundata, {
            "step": self.step_counter,
            "gradient_norm": np.linalg.norm(dKL),
            "mean": np.mean(particles),
            "std": np.std(particles),
        })

    @partial(jit, static_argnums=0)
    def transformation(self, x, particles, step_counter):
        # inject x
        particles_with_inject = index_update(particles, index[0], x)
        optimizer_state_with_inject = self.opt.init(particles_with_inject)

        # step
        updated_optimizer_state, *_ = self._step(
            optimizer_state_with_inject, >>>>>> ,step_counter)

        # extract z = T(x)
        updated_particles = self.opt.get_params(updated_optimizer_state)
        z = updated_particles[follower_idx[0]]
        return z

    def update_logp(self):
        def t(x):
            return self.transformation(x, self.get_params(), self.step_counter)
        self.pullback_logp = metrics._pushforward_log(self.pullback_logp, t)

    def logp_value_and_grad(self):
        return vmap(value_and_grad(self.pullback_logp))(self.get_params())

    def flow(self, key=None, n_iter=400):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        for i in tqdm(range(n_iter), disable=disable_tqdm):
            self.step()
            key, subkey = random.split(key)


proposal = distributions.Gaussian(0, 5)
target = distributions.Gaussian(0, 1)


key = random.PRNGKey(0)
k = KLFlow(key,
          target=target,
          proposal=proposal)


k.flow()
