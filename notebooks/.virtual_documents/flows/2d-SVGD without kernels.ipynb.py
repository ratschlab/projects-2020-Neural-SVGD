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


# d = 2
# coords = np.vstack([np.eye(d), np.ones(d).reshape(1, d)]) * 10

# # sample m points from the corners of n-simplex

# m = 3
# idx = random.choice(key, d+1, (m,), replace=False)
# means = coords[idx]

# target = distributions.GaussianMixture(means, 1, np.ones(m))
# proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
# setup = distributions.Setup(target, proposal)
setup = distributions.squiggle_target
target, proposal = setup.get()


get_ipython().run_line_magic("autoreload", "")


n_steps = 200
n_particles = 300
particle_lr = 1e-2
learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1., sizes=[32, 32, 2], patience=20,
                                                               learner_lr=learner_lr)
# svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=100,         n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(neural_learner.rundata["training_loss"], "--.", label="Trainging Loss")
plt.plot(neural_learner.rundata["validation_loss"], "--.", label="Validation Loss")
plt.legend()


get_ipython().run_line_magic("matplotlib", " inline")
particles = neural_particles.get_params()
fig, ax = plt.subplots(figsize=[10, 10])
a=2
lims = (-4, 4)
ax.set(xlim=lims, ylim=lims)

plot.plot_fun_2d(target.pdf, lims=lims, ax=ax, alpha=0.5)
plot.plot_gradient_field(utils.negative(neural_learner.grads), ax, lims=lims)
plot.scatter(particles, ax=ax)


neural_trajectories = np.asarray(neural_particles.rundata["particles"])
svgd_trajectories = np.asarray(svgd_particles.rundata["particles"])
sgld_trajectories = np.asarray(sgld_particles.rundata["particles"])


neural_trajectories, sgld_trajectories, svgd_trajectories = [traj[:, p.group_idx[0], :] for traj, p in zip([neural_trajectories, sgld_trajectories, svgd_trajectories], 
                                                                                                           [neural_particles, sgld_particles, svgd_particles])]


get_ipython().run_line_magic("matplotlib", " inline")
fig, axs = plt.subplots(1, 3, figsize=[25, 8])
axs = axs.flatten()
lim = (-4, 4)

for ax in axs:
    ax.set(xlim=lim, ylim=lim)
ax1 = neural_particles.plot_final(ax=axs[0], target=target, cmap="Greens")
ax1.set_title("Neural SVGD")

ax2 = sgld_particles.plot_final(ax=axs[1], target=target, cmap="Greens")
ax2.set_title("SGLD")

ax = axs[2]
plot.plot_fun_2d(target.pdf, lims=lim, ax=ax, cmap="Greens")
plot.scatter(target.sample(300), ax=ax, color="tab:orange")
ax.set_title("True Samples")


get_ipython().run_line_magic("matplotlib", " widget")
fig, axs = plt.subplots(1, 2, figsize=[14, 6])
axs=axs.flatten()
for ax in axs:
    ax.set(xlim=lim, ylim=lim)

animations = []
for ax, trajectories, title, i in zip(axs, [neural_trajectories, sgld_trajectories], ["Neural", "SGLD"], [10, 10]):
    ax.set_title(title)
    plot.plot_fun_2d(target.pdf, lims=lim, ax=ax)
    animations.append(plot.animate_array(trajectories, fig, ax, interval=i))






