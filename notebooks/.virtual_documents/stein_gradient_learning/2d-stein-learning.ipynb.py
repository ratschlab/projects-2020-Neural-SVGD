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
import kernel_learning
import discrepancy_learner

from jax.experimental import optimizers

key = random.PRNGKey(0)


@partial(jit, static_argnums=1)
def get_sd(samples, fun):
    return stein.stein_discrepancy(samples, target.logpdf, fun)


setup = distributions.banana_proposal
target, proposal = setup.get()
sizes = [32, 32, 16, 2]

setup.plot()


def get_ksds(proposal, kernel):
    @jit
    def compute_ksd(samples):
        return stein.ksd_squared_u(samples, target.logpdf, kernel)
    ksds = []
    for _ in tqdm(range(100)):
        samples = proposal.sample(400)
        ksds.append(compute_ksd(samples))
    return ksds


learning_rate = 1e-2
learner = discrepancy_learner.SDLearner(key,
                                        target,
                                        sizes,
                                        learning_rate,
                                        lambda_reg=1)


samples = proposal.sample(400)
learner.train(samples, n_steps=10**3, proposal=proposal)


# compute optimal KSD
div = 1 if learner.lambda_reg == 0 else 2*learner.lambda_reg
def kl_gradient(x):
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x) / div
sds = []
for _ in range(100):
    samples = proposal.sample(400)
    sds.append(get_sd(samples, kl_gradient))


fig, axs = plt.subplots(figsize=[10, 8])
plt.plot(learner.rundata["training_sd"], label="Stein Discrepancy")
plt.errorbar(x=0, y=onp.mean(sds), yerr=onp.std(sds), fmt="o", capsize=10, color="green")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal SD", color="green")
plt.legend()


plt.plot(learner.rundata["fnorm"])
plt.yscale("log")


learned_stein_gradient = jit(learner.get_f())
rbf_stein_gradient = jit(stein.get_phistar(kernels.get_rbf_kernel(.1), target.logpdf, samples))


fig, axs = plt.subplots(1, 3, figsize=[26,8])
axs = axs.flatten()
def log_diff(x):
    return (target.logpdf(x) - proposal.logpdf(x)) / div

samples = None
samples = proposal.sample(100)
xlims = [-6, 6]
ylims = [-5, 10]
fields = (kl_gradient, learned_stein_gradient, rbf_stein_gradient)
scales = (20, 20, 0.2)
for ax, vector_field, scale in zip(axs, fields, scales):
#     ax.pcolormesh(xx, yy, zz, vmin=-50, vmax=1, cmap="Blues")
    setup.plot(ax=ax, xlims=xlims, ylims=ylims)
    plot.quiverplot(vector_field, samples=samples, num_gridpoints=10, scale=scale, xlims=xlims, ylims=ylims, ax=ax)
    ax.set_ylim(ylims)


fig, axs = plt.subplots(1, 3, figsize=[26,8])
axs = axs.flatten()
def log_diff(x):
    return (target.logpdf(x) - proposal.logpdf(x)) / div

xx, yy, zz = plot.make_meshgrid(log_diff, (-15, 15))
xlims = [-5, 5]
ylims = [-10, 10]

fields = (kl_gradient, learned_stein_gradient, rbf_stein_gradient)
scales = (100, 100, 1)
for ax, vector_field, scale in zip(axs, fields, scales):
#     ax.pcolormesh(xx, yy, zz, vmin=-50, vmax=1, cmap="Blues")
    setup.plot(ax=ax, lims=(-10, 10))
    plot.quiverplot(vector_field, num_gridpoints=10, scale=scale, xlims=xlims, ylims=ylims, ax=ax)


stein_gradient = learned_stein_gradient
def phi_norm(x): return np.linalg.norm(stein_gradient(x))**2
np.mean(vmap(phi_norm)(samples))


kl_gradient
def dkl_norm(x): return np.linalg.norm(kl_gradient(x))**2
np.mean(vmap(dkl_norm)(samples))


@jit
def get_norm(key):
    s = proposal.sample(1000, key=key)
    return np.mean(vmap(dkl_norm)(s))

norms=[]
for key in random.split(key, 100):
    norms.append(get_norm(key))


m = onp.mean(norms)


plt.plot(learner.rundata["fnorm"])
plt.axhline(m)
plt.yscale("log")
