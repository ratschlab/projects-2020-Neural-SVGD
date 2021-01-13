# Maximize the stein discrepancy, keeping distributions p and q fixed. Compare
# neural stein discrepancy, kernelized SD, and theoretical optimum.
import os
from functools import partial
from jax import grad, jit, random
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import json_tricks as json

import utils
import stein
import kernels
import distributions
import models
import config as cfg

from jax.experimental import optimizers

key = random.PRNGKey(0)

# Config
# set up exporting
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'pgf.rcfonts': False,
    'axes.unicode_minus': False, # avoid unicode error on saving plots with negative numbers (??)
})

#figure_path = "/home/lauro/documents/msc-thesis/paper/latex/figures/"


# Poorly conditioned Gaussian
d = 50
variances = jnp.logspace(-5, 0, num=d)
target = distributions.Gaussian(jnp.zeros(d), variances)
proposal = distributions.Gaussian(jnp.zeros(d), jnp.ones(d))


print("Computing theoretically optimal Stein discrepancy...")
@partial(jit, static_argnums=1)
def get_sd(samples, fun):
    """Compute SD(samples, p) given witness function fun"""
    return stein.stein_discrepancy(samples, target.logpdf, fun)

def kl_gradient(x):
    """Optimal witness function."""
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x) / 2
sds = []
for _ in range(100):
    samples = proposal.sample(100)
    sds.append(get_sd(samples, kl_gradient))


# Kernelized SD (using Gaussian kernel w/ median heuristic)
print("Computing kernelized Stein discrepancy...")
@jit
def compute_scaled_ksd(samples):
    kernel = kernels.get_rbf_kernel(kernels.median_heuristic(samples))
    phi = stein.get_phistar(kernel, target.logpdf, samples)
    ksd = stein.stein_discrepancy(samples, target.logpdf, phi)
    return ksd**2 / utils.l2_norm_squared(samples, phi)
ksds = []
for _ in range(100):
    samples = proposal.sample(400)
    ksds.append(compute_scaled_ksd(samples))


# Neural SD
print("Computing neural Stein discrepancy...")
key, subkey = random.split(key)
learner = models.SDLearner(target_dim=d,
                           target_logp=target.logpdf,
                           key=subkey,
                           learning_rate=1e-2,
                           patience=-1)
BATCH_SIZE=1000
def sample(key):
    return proposal.sample(BATCH_SIZE*2, key).split(2)

key, subkey = random.split(key)
learner.train(next_batch=sample, key=subkey, n_steps=800, progress_bar=True)
learner.done()





print("Saving results...")
# Plotting config
plt.rc('font', size=7)
plt.legend(fontsize='small')
printsize_singlecolumn = [3.6, 3] # adapted for ICML submission (single-column)

fig, axs = plt.subplots(figsize=printsize_singlecolumn)

plt.plot(learner.rundata["training_sd"], label="Stein discrepancy given $f_{\\theta}$")
plt.axhline(y=onp.mean(sds), linestyle="--", label="Optimal Stein discrepancy", color="green")
plt.axhline(y=onp.mean(ksds), linestyle="--", label="Kernelized Stein discrepancy", color="tab:orange")

plt.ylabel("Stein discrepancy")
plt.xlabel("Iteration")

plt.savefig(cfg.figure_path + "sd_maxing.pgf")


# save json results
results = {
    "KSD": onp.mean(ksds).tolist(),
    "Optimal_SD": onp.mean(sds).tolist(),
    "Neural_SD": learner.rundata["training_sd"].tolist(),
}

with open(results_path + "sd_maxing.json", "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)


# save json results
results = {
    "KSD": onp.mean(ksds).tolist(),
    "Optimal_SD": onp.mean(sds).tolist(),
    "Neural_SD": learner.rundata["training_sd"].tolist(),
}

with open(cfg.results_path + "sd_maxing.json", "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)
