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

from jax.experimental import optimizers

key = random.PRNGKey(43)


get_ipython().run_line_magic("autoreload", "")


# setup = distributions.double_mixture
# target, proposal = setup.get()
target = distributions.GaussianMixture([-3, 3], [1, 1], [1/3, 2/3])
proposal = distributions.Gaussian(-5, 1)
setup = distributions.Setup(target, proposal)
sizes = [32, 32, 1]
learning_rate = 1e-2
particle_lr = 1e-1
lambda_reg = 1
n_particles = 1000 # (num samples)
sample_every = False
noise_level = 0.0


def phistar_batched(params, _, particles, aux=False):
    kernel = kernels.get_rbf_kernel(1)
    inducing_particles = params
    phi = stein.get_phistar(kernel, target.logpdf, inducing_particles)
    l2_phi_squared = utils.l2_norm(inducing_particles, phi)**2
    ksd = stein.stein_discrepancy(inducing_particles, target.logpdf, phi)
    alpha = ksd / (2*lambda_reg*l2_phi_squared)
    if aux:
        return -alpha*vmap(phi)(particles), {"ksd": ksd, "alpha": alpha}
    else:
        return -alpha*vmap(phi)(particles)


def phistar_batched_adaptive(params, _, particles, aux=False):
    inducing_particles = params
    bandwidth = kernels.median_heuristic(inducing_particles)
    kernel = kernels.get_rbf_kernel(bandwidth)
    phi = stein.get_phistar(kernel, target.logpdf, inducing_particles)
    l2_phi_squared = utils.l2_norm(inducing_particles, phi)**2
    ksd = stein.stein_discrepancy(inducing_particles, target.logpdf, phi)
    alpha = ksd / (2*lambda_reg*l2_phi_squared)
    if aux:
        return -alpha*vmap(phi)(particles), {"ksd": ksd, "alpha": alpha}
    else:
        return -alpha*vmap(phi)(particles)


get_ipython().run_line_magic("autoreload", "")


key, subkey = random.split(key)
stein_learner = models.SDLearner(subkey,
                                 target=target,
                                 sizes=sizes,
                                 learning_rate=learning_rate,
                                 lambda_reg=lambda_reg)
key, subkey = random.split(key)
particles_score = models.Particles(subkey,
                             gradient=stein_learner.kl_gradient,
                             proposal=proposal,
                             n_particles=n_particles,
                             learning_rate=particle_lr,
                             only_training=True)

particles_svgd = models.Particles(subkey,
                             gradient=phistar_batched,
                             proposal=proposal,
                             n_particles=n_particles,
                             learning_rate=particle_lr,
                             only_training=True)


def step_schedule(step_count):
    """Return nr of stein learner iterations
    at (particle) step step_count"""
    if step_count < 2:
        return 150
    else:
        return 50 if step_count < 10 else 10


n_steps=1000
n_float_errs = 0
for i in tqdm(range(n_steps)):
    n_learner_steps = step_schedule(i)
    train_x, val_x = particles_score.get_params(split_by_group=True)
    try:
        x = stein_learner.train(train_x, val_x, key=subkey, n_steps=n_learner_steps, noise_level=noise_level)
    except FloatingPointError:
        n_float_errs += 1
        particles_score.perturb()
        ...
    particles_score.step(stein_learner.get_params())

    inducing_particles, _ = particles_svgd.get_params(split_by_group=True)
    particles_svgd.step(inducing_particles)


ylim1 = (-10, 1.5)
ylim2 = (1, 3.5)
# ylim1 = (-0.5, 1.5)
# ylim2 = (4.5, 5.6)
# ylim1=None
# ylim2=None

fig, axs = plt.subplots(1, 2, figsize=[18,5])
axs = axs.flatten()

ax = axs[0]
ax.plot(particles_score.rundata["training_mean"], label="SVGD-learned mean")
ax.axhline(y=target.mean, linestyle="--", color="green")
ax.set_ylim(ylim1)
ax.plot(particles_svgd.rundata["training_mean"], label="SVGD mean")
ax.axhline(y=target.mean, linestyle="--", color="green")
ax.set_ylim(ylim1)
ax.legend()

ax = axs[1]
ax.plot(particles_score.rundata["training_std"], label="SVGD-learned std")
# ax.plot(particles_score.rundata["validation_std"])
ax.axhline(y=np.sqrt(target.cov), linestyle="--", color="green")
ax.set_ylim(ylim2)
ax.plot(particles_svgd.rundata["training_std"], label="SVGD std")
ax.axhline(y=np.sqrt(target.cov), linestyle="--", color="green")
ax.set_ylim(ylim2)
ax.legend()


fig, axs = plt.subplots(1, 2, figsize=[16, 6])
axs = axs.flatten()
x_train_sl, single_val = particles_score.get_params(split_by_group=True)
x_train_svgd, _ = particles_svgd.get_params(split_by_group=True)

for ax, x_train in zip(axs, (x_train_sl, x_train_svgd)):
    ax.hist(x_train[:,0], density=True, alpha=0.5, bins=25)
    plot.plot_fun(target.pdf, ax=ax, lims=(-15, 14))


plt.plot(stein_learner.rundata["fnorm"])
plt.ylim(-0.1, 1.2)



