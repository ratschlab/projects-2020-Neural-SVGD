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

key = random.PRNGKey(0)


# set up exporting
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': False,
    'pgf.rcfonts': False,
})

figure_path = "/home/lauro/documents/msc-thesis/thesis/figures/"
# save figures by using plt.savefig('title of figure')
# remember that latex textwidth is 5.4in
# so use figsize=[5.4, 4], for example


get_ipython().run_line_magic("autoreload", "")


d = 50
variances = np.logspace(-5, 0, num=d)
target = distributions.Gaussian(np.zeros(d), variances)
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
setup = distributions.Setup(target, proposal)


@partial(jit, static_argnums=1)
def get_sd(samples, fun):
    return stein.stein_discrepancy(samples, target.logpdf, fun)


def get_ksds(proposal, kernel):
    @jit
    def compute_ksd(samples):
        return stein.ksd_squared_u(samples, target.logpdf, kernel)
    ksds = []
    for _ in tqdm(range(100)):
        samples = proposal.sample(400)
        ksds.append(compute_ksd(samples))
    return ksds


get_ipython().run_line_magic("autoreload", "")


learning_rate = 1e-2
key, subkey = random.split(key)
learner = models.SDLearner(target_dim=d,
                           target_logp=target.logpdf,
                           key=subkey,
                           learning_rate=learning_rate,
                           patience=-1)

batch_size=1000


def sample(key):
    return proposal.sample(batch_size*2, key).split(2)
key, subkey = random.split(key)
learner.train(next_batch=sample, key=subkey, n_steps=150, progress_bar=True)


# compute optimal SD
div = 2*learner.lambda_reg
def kl_gradient(x):
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x) / div
sds = []
for _ in range(100):
    samples = proposal.sample(400)
    sds.append(get_sd(samples, kl_gradient))


samples = proposal.sample(1000)
utils.l2_norm_squared(samples, kl_gradient)


printsize = [5.4, 4]
showsize = [15, 8]


get_ipython().run_line_magic("matplotlib", " inline")


fig, axs = plt.subplots(figsize=showsize) # remember that latex textwidth is 5.4in

plt.plot(learner.rundata["training_sd"], label="Stein Discrepancy given $f_{\theta}$")
# plt.plot(learner.rundata["training_loss"], label="Stein Discrepancy given $f_{\theta}$")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal Stein Discrepancy", color="green")
plt.legend()


fig, axs = plt.subplots(figsize=printsize) # remember that latex textwidth is 5.4in

plt.plot(learner.rundata["training_sd"], label="Stein Discrepancy given $f_{\\theta}$")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal Stein Discrepancy", color="green")
plt.legend()


plt.savefig(figure_path + "sd_maxing.pgf")
plt.savefig("sd_maxing.pgf")


d = 50
key, subkey = random.split(key)
alpha = 0.5
beta = 1
variances = random.gamma(subkey, alpha, shape=(d,)) / beta

target = distributions.Gaussian(np.zeros(d), variances)
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
setup = distributions.Setup(target, proposal)


plt.hist(variances)


plt.hist(np.log(variances))


def get_ksds(proposal, kernel):
    @jit
    def compute_ksd(samples):
        return stein.ksd_squared_u(samples, target.logpdf, kernel)
    ksds = []
    for _ in tqdm(range(100)):
        samples = proposal.sample(400)
        ksds.append(compute_ksd(samples))
    return ksds


get_ipython().run_line_magic("autoreload", "")


learning_rate = 1e-2
key, subkey = random.split(key)
learner = models.SDLearner(target_dim=d,
                           target_logp=target.logpdf,
                           key=subkey,
                           learning_rate=learning_rate,
                           patience=-1)

batch_size=1000


def sample(key):
    return proposal.sample(batch_size*2, key).split(2)
key, subkey = random.split(key)
learner.train(next_batch=sample, key=subkey, n_steps=1000, progress_bar=True)


# compute optimal SD
div = 2*learner.lambda_reg
def kl_gradient(x):
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x) / div
sds = []
for _ in range(100):
    samples = proposal.sample(400)
    sds.append(get_sd(samples, kl_gradient))


samples = proposal.sample(1000)
utils.l2_norm_squared(samples, kl_gradient)


printsize = [5.4, 4]
showsize = [15, 8]


get_ipython().run_line_magic("matplotlib", " inline")


fig, axs = plt.subplots(figsize=showsize) # remember that latex textwidth is 5.4in

plt.plot(learner.rundata["training_sd"], label="Stein Discrepancy given $f_{\theta}$")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal Stein Discrepancy", color="green")
plt.legend()


fig, axs = plt.subplots(figsize=printsize) # remember that latex textwidth is 5.4in

plt.plot(learner.rundata["training_sd"], label="Stein Discrepancy given $f_{\theta}$")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal Stein Discrepancy", color="green")
plt.legend()


plt.savefig(figure_path + "sd_maxing.pgf")
plt.savefig("sd_maxing.pgf")


d = 1000
key, subkey = random.split(key)
alpha = 0.5
beta = 1
variances = random.gamma(subkey, alpha, shape=(d,)) / beta

target = distributions.Gaussian(np.zeros(d), variances)
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
setup = distributions.Setup(target, proposal)


get_ipython().run_line_magic("matplotlib", " inline")


plt.hist(np.log(variances))


learning_rate = 1e-2
key, subkey = random.split(key)
learner = models.SDLearner(target_dim=d,
                           target_logp=target.logpdf,
                           key=subkey,
                           learning_rate=learning_rate,
                           patience=-1,
                           sizes=[256, 256, 1000])

batch_size=10


def sample(key):
    return proposal.sample(batch_size*2, key).split(2)
key, subkey = random.split(key)
learner.train(next_batch=sample, key=subkey, n_steps=2000, progress_bar=True)


# compute optimal SD
div = 2*learner.lambda_reg
def kl_gradient(x):
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x) / div
sds = []
for _ in range(100):
    samples = proposal.sample(40)
    sds.append(get_sd(samples, kl_gradient))


samples = proposal.sample(40)
utils.l2_norm_squared(samples, kl_gradient)


def get_ksds(kernel):
    @jit
    def compute_ksd(samples):
        return stein.ksd_squared_u(samples, target.logpdf, kernel)
    ksds = []
    for _ in tqdm(range(100)):
        samples = proposal.sample(10)
        ksds.append(compute_ksd(samples))
    return ksds


ksds = get_ksds(kernels.get_rbf_kernel(1.))


samples = proposal.sample(100)
median_h = kernels.median_heuristic(samples)
median_heuristic_ksds = get_ksds(kernels.get_rbf_kernel(median_h))


printsize = [5.4, 4]
showsize = [15, 8]
fig, axs = plt.subplots(figsize=showsize) # remember that latex textwidth is 5.4in

plt.plot(learner.rundata["training_sd"], label="Stein Discrepancy given $f_{\\theta}$")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal Stein Discrepancy", color="green")
plt.axhline(y=onp.mean(ksds), linestyle="--", label="Kernel Stein Discrepancy", color="tab:orange")
plt.axhline(y=onp.mean(median_heuristic_ksds), linestyle="--", label="Kernel Stein Discrepancy", color="tab:orange")
plt.legend()


# plt.savefig(figure_path + "sd_maxing.pgf")
# plt.savefig("sd_maxing.pgf")



