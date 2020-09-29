get_ipython().run_line_magic("load_ext", " autoreload")
from jax import config
config.update("jax_debug_nans", True)
config.update("jax_disable_jit", False)

import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
import json_tricks as json
import copy
from functools import partial
import warnings

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


get_ipython().run_line_magic("matplotlib", " inline")
# setup = distributions.ring_target
# target, proposal = setup.get()
target = distributions.Banana([0, 0], [4, 1])
proposal = distributions.Gaussian([-5, -5], 1)
setup = distributions.Setup(target, proposal)
setup.plot(lims=(-15, 15))


get_ipython().run_line_magic("autoreload", "")


n_steps = 600
noise = 1.
# key, subkey = random.split(key)
# learner, neural_particles, err =        deep_kernel_flow(subkey, setup, n_steps=n_steps, particle_lr=1e-2, noise_level=noise, sizes=[32, 32, 2])
kernel_gradient, kernel_particles, err = flows.svgd_flow(subkey, setup, n_steps=n_steps, particle_lr=1e-2, noise_level=0)


get_ipython().run_line_magic("matplotlib", " inline")
n_steps = learner.rundata["train_steps"]
plt.plot(n_steps)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(learner.rundata["training_loss"], "--.", label="Trainging Loss")
plt.plot(learner.rundata["validation_loss"], "--.", label="Validation Loss")
plt.legend()


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(learner.rundata["training_ksd"], "--.", label="Trainging KSD")
plt.plot(learner.rundata["validation_ksd"], "--.", label="Validation KSD")
plt.legend()


learner.rundata.keys()


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(learner.rundata["training_l2_norm"], "--.", label="L2 squared")
plt.plot(learner.rundata["validation_l2_norm"], "--.", label="val")
plt.yscale("log")
plt.legend()


neural_trajectories = np.asarray(neural_particles.rundata["particles"])
kernel_trajectories = np.asarray(kernel_particles.rundata["particles"])


get_ipython().run_line_magic("matplotlib", " widget")
fig, axs = plt.subplots(1, 2, figsize=[14, 6])
axs=axs.flatten()
lim=(-10, 10)
for ax in axs:
    ax.set(xlim=lim, ylim=lim)

animations = []
for ax, trajectories, title in zip(axs, [neural_trajectories, kernel_trajectories], ["Neural", "Kernel"]):
    ax.set_title(title)
    plot.plot_fun_2d(target.pdf, lims=(-13, 13), ax=ax)
    animations.append(plot.animate_array(trajectories, fig, ax))


get_ipython().run_line_magic("matplotlib", " inline")
fig, axs = plt.subplots(1, 3, figsize=[25, 8])
axs = axs.flatten()
for ax in axs:
    ax.set(xlim=lim, ylim=lim)
ax1 = neural_particles.plot_final(ax=axs[0], target=target, cmap="Greens")
ax1.set_title("Neural SVGD")

ax2 = kernel_particles.plot_final(ax=axs[1], target=target, cmap="Greens")
ax2.set_title("SVGD")

ax = axs[2]
plot.plot_fun_2d(target.pdf, lims=(-13, 13), ax=ax, cmap="Greens")
plot.scatter(target.sample(300), ax=ax, color="tab:orange")
ax.set_title("True Samples")






