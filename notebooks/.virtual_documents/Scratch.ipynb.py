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
config.update("jax_debug_nans", True)
# config.update("jax_log_compiles", True)
# config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import optax
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
import models

from jax.experimental import optimizers

key, subkey = random.split(random.PRNGKey(0))

from jax.scipy.stats import norm


get_ipython().run_line_magic("autoreload", "")


target = distributions.Gaussian(0, 1)
proposal = distributions.Gaussian(-3, 1)

key, subkey = random.split(key)
gradient = models.EnergyGradient(target, subkey)
key, subkey = random.split(key)
particles = models.Particles(subkey, gradient.gradient, proposal, n_particles=50, num_groups=1, learning_rate=1e-1, optimizer="adam", noise_level=1.)


scales = []
for _ in range(100):
    particles.step(None)
    scales.append(onp.squeeze(onp.abs(particles.noise_scales)).tolist())


fig, axs = plt.subplots(2, figsize=[8, 6])
particles.plot_trajectories(marker=".", ax=axs[0])
axs[1].plot(scales, "--.")
plt.yscale("log")


fig, axs = plt.subplots(2, figsize=[8, 6])
particles.plot_trajectories(marker=".", ax=axs[0])
axs[1].plot(scales, "--.")
plt.yscale("log")


from distributions import Gaussian as G


get_ipython().run_line_magic("autoreload", "")


target = G(0,1)


learner = models.SDLearner(target)
particles = models.Particles(key, learner.gradient, target)


learner.train(particles.next_batch, key=subkey, n_steps=5)
particles.step(learner.get_params())


learner.rundata["train_steps"]


learner.step_counter


learner.rundata["step_counter"]


get_ipython().run_line_magic("autoreload", "")


def check_dist(dist, key):
    s = dist.sample(10**4, key=key)
    errs = {
        "se_mean": np.sum((np.mean(s, axis=0) - dist.mean)**2),
        "se_cov": np.sum((np.cov(s, rowvar=False) - dist.cov)**2),
    }
    for k, err in errs.items():
        if err > 1e-2:
            print(f"{k} to big! {k} = {err} > 0.01")
    return


dist = distributions.Gaussian(0, 1)
check_dist(dist, key)


dist, _ = distributions.funnel.get()
check_dist(dist, key) # fine, just hard to approximate


dist, _ = distributions.banana_target.get()
check_dist(dist, key) # fine


dist, _ = distributions.ring_target.get()
check_dist(dist, key) # fine


dist, _ = distributions.squiggle_target.get()
check_dist(dist, key)



