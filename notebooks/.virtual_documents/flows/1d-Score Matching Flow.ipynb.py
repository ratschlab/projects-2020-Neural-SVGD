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
kernel_gradient, kernel_particles, err = flows.score_flow(subkey, setup)


fig, axs = plt.subplots(2, 2, figsize=[20, 10])
kernel_particles.plot_mean_and_std(target, axs=axs[0])
neural_score_particles.plot_mean_and_std(target, axs=axs[1])


fig, axs = plt.subplots(2, figsize=[20, 10])
neural_score_particles.plot_trajectories(ax=axs[0])
kernel_particles.plot_trajectories(ax=axs[1])


fig, axs = plt.subplots(2, figsize=[20, 10])
lim = (-10, 10)
for ax in axs: ax.set_xlim(lim)
neural_score_particles.plot_final(ax=axs[0], target=target, lims=lim)
kernel_particles.plot_final(ax=axs[1], target=target, lims=lim)
