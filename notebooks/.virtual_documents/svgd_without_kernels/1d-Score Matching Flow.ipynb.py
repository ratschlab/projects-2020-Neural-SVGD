get_ipython().run_line_magic("load_ext", " autoreload")
from jax import config
config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
import json_tricks as json
import copy
from functools import partial

from tqdm import tqdm
import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import matplotlib.pyplot as plt
import numpy as onp
import jax
import pandas as pd
import haiku as hk
from jax.experimental import optimizers

import config

import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import models
import flows

from jax.experimental import optimizers

key = random.PRNGKey(0)


# setup = distributions.double_mixture
# target, proposal = setup.get()
target = distributions.GaussianMixture([-3, 3], [1, 1], [1/3, 2/3])
proposal = distributions.Gaussian(-5, 1)
setup = distributions.Setup(target, proposal)

setup.plot()


key, subkey = random.split(key)
neural_score_learner, neural_score_particles, err = flows.neural_score_flow(subkey, setup)


kernel_gradient = models.KernelizedScoreMatcher(target=target,
                                                key=key,
                                                lambda_reg=1/2)


s = proposal.sample(100)


score = kernel_gradient.get_score(proposal.sample(100))


plot.plot_fun(utils.reshape_input(score), lims=(-10,5))
plot.plot_fun(grad(proposal.pdf), lims=(-10, 5))
plt.scatter(s, vmap(score)(s))


x = np.array([1.])
score(x)


def score_flow(key, setup, n_particles=300, n_steps=100, particle_lr=1e-1, lambda_reg=1/2, noise_level=0):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()

    kernel_gradient = models.KernelizedScoreMatcher(target=target,
                                                    key=keya,
                                                    lambda_reg=lambda_reg)
    svgd_particles = models.Particles(key=keyb,
                                     gradient=kernel_gradient.gradient,
                                     proposal=proposal,
                                     n_particles=n_particles,
                                     learning_rate=particle_lr)

    for _ in tqdm(range(n_steps)):
        svgd_particles.step(None, noise_pre=noise_level)

    return kernel_gradient, svgd_particles, None

key, subkey = random.split(key)
kernel_gradient, kernel_particles, err = score_flow(subkey, setup)


fig, axs = plt.subplots(2, 2, figsize=[20, 10])
svgd_particles.plot_mean_and_std(target, axs=axs[0])
neural_svgd_particles.plot_mean_and_std(target, axs=axs[1])


fig, axs = plt.subplots(2, figsize=[20, 10])
neural_svgd_particles.plot_trajectories(ax=axs[0])
# svgd_particles.plot_trajectories(ax=axs[1])


fig, axs = plt.subplots(2, figsize=[20, 10])
neural_svgd_particles.plot_final(ax=axs[0], target=target)
svgd_particles.plot_final(ax=axs[1], target=target)


# particles = score_particles
particles = svgd_particles

# learner = score_learner
# learner = svgd_learner
grad = kernel_gradient

samples = proposal.sample(1000)
v = kernel_gradient.get_field(samples)
# v = learner.get_field(samples)
# score = learner.get_score(samples)
# v_true = setup.grad_kl


# @utils.reshape_input
# def dlogp(x):
#     return score(x) - v(x)

fig, ax = plt.subplots(figsize=[18, 6])
plot.plot_fun(utils.reshape_input(v),     lims=(-8, 8), label="learned grad(kl)")
# plot.plot_fun(utils.reshape_input(score), lims=(-8, 8), label="learned score")
plt.axhline(y=0, linestyle="--")
# plot.plot_fun(utils.negative(grad(target.logpdf)), lims=(-8, 8), label="-grad(logp)")
ax.set_xlim(xlims)
ax.legend()



