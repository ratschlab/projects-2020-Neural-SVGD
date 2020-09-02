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

from jax.experimental import optimizers

key = random.PRNGKey(0)


# proposal = distributions.Gaussian([0]*50, 1)
# target = distributions.Gaussian([2]*50, 1)

proposal = distributions.Gaussian(0, 5)
target = distributions.GaussianMixture([-3, 0, 3], [1, 0.05, 1], [1,1,1])
sizes = [1]

plot.plot_fun(proposal.pdf)
plot.plot_fun(target.pdf, label="Target")


def get_ksds(proposal, kernel):
    @jit
    def compute_ksd(samples):
        return stein.ksd_squared_u(samples, target.logpdf, kernel)
    ksds = []
    for _ in tqdm(range(100)):
        samples = proposal.sample(400)
        ksds.append(compute_ksd(samples))
    return ksds


learning_rate = 0.05
learner = kernel_learning.KernelLearner(key,
                                        target,
                                        sizes,
                                        kernels.get_rbf_kernel(1),
                                        learning_rate,
                                        lambda_reg=1,
                                        scaling_parameter=True)
kernel = learner.get_kernel(learner.get_params())
ksds_pre = get_ksds(proposal, kernel)


print("Training kernel to optimize KSD...")
samples = proposal.sample(400)
learner.train(samples, n_steps=500)


ksds_post = get_ksds(proposal, learner.get_kernel(learner.get_params()))


print("Plot results:")

rundata = learner.rundata
fig, axs = plt.subplots(1, 2, figsize=[18,5])
axs = axs.flatten()
axs[0].plot(rundata["ksd_squared"], "--", label="KSD")
axs[0].errorbar(x=0, y=onp.mean(ksds_pre), yerr=onp.std(ksds_pre), fmt="o", capsize=10)
axs[0].errorbar(x=len(rundata["loss"]), y=onp.mean(ksds_post), yerr=onp.std(ksds_post), fmt="o", capsize=10)
axs[0].legend()

axs[1].plot(rundata["bandwidth"])
axs[1].set_yscale("log")


samples = proposal.sample(500)
phistar = learner.get_phistar(learner.get_params(), samples)
def optimal_phistar(x):
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x)
phistar_rbf = stein.get_phistar(kernels.get_rbf_kernel(1), target.logpdf, samples)


div = 1 if learner.lambda_reg == 0 else 2*learner.lambda_reg
# plot the stein gradient
if target.d ==1:
    grid_n = 100
    fig, ax = plt.subplots(figsize=[12,7])

    grid = np.linspace(-5, 5, grid_n).reshape(grid_n, 1)
    plt.plot(grid, vmap(optimal_phistar)(grid)/div, label="KL gradient \\nabla logp/logq")
    plt.plot(grid, vmap(phistar)(grid), label="learned_phistar")
    plt.plot(grid, vmap(phistar_rbf)(grid)*rundata["normalizing_const"][-1], label="rbf phistar")

    plt.legend()


stein.globally_maximal_stein_discrepancy(proposal, target)


params = learner.opt.get_params(learner.optimizer_state)
learned_kernel = learner.get_kernel(params)


s = proposal.sample(100)
learned_kernel(s[4], s[10])


if proposal.d == 1:
    ngrid = 10**4
    grid = np.linspace(-4, 10, ngrid).reshape(ngrid,1)
    x = np.array([0.])
    plt.plot(grid, vmap(learned_kernel, (0, None))(grid, x), label="Learned", color="r")
    plt.plot(grid, vmap(kernels.get_rbf_kernel(1),  (0, None))(grid, x), label="RBF", color="b")
    x = np.array([4.])
    plt.plot(grid, vmap(learned_kernel, (0, None))(grid, x), color="r")
    plt.plot(grid, vmap(kernels.get_rbf_kernel(1),  (0, None))(grid, x), color="b")

    plt.legend()
elif proposal.d == 2:
    fig, ax = plt.subplots(figsize=[7,7])
    x = np.array([0, 0])
    def kernfunx(x_): return learned_kernel(x, x_)
#     def rbfx(x_): return kernels.get_rbf_kernel(1)(x, x_)
    plot.plot_pdf_2d(kernfunx, lims=(-10, 10), label="Learned Kernel", ax=ax)
#     plot.plot_pdf_2d(rbfx, lims=(-10, 10), label="RBF", ax=ax)
