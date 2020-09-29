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


get_ipython().run_line_magic("autoreload", "")


get_ipython().run_line_magic("matplotlib", " inline")
# setup = distributions.funnel
setup = distributions.mix_of_gauss
target, proposal = setup.get()
setup.plot(lims=(-15, 15))


n_steps = 1000
noise = 1.
particle_lr = 1e-2

key, subkey = random.split(key)


sgld_gradient, sgld_particles, err = flows.sgld_flow(subkey, setup, n_steps=n_steps, particle_lr=particle_lr, noise_level=1., n_particles=n_particles, particle_optimizer="adam")
svgd_gradient, svgd_particles, err = flows.svgd_flow(subkey, setup, n_steps=n_steps, particle_lr=particle_lr, noise_level=1., n_particles=n_particles)


sgld_trajectories = np.asarray(sgld_particles.rundata["particles"])
svgd_trajectories = np.asarray(svgd_particles.rundata["particles"])


sgld_trajectories, svgd_trajectories = [traj[:, p.group_idx[0], :] for traj, p in zip([sgld_trajectories, svgd_trajectories], [particles, svgd_particles])]


get_ipython().run_line_magic("matplotlib", " widget")
fig, axs = plt.subplots(1, 2, figsize=[14, 6])
axs=axs.flatten()
lim=(-7, 7)
for ax in axs:
    ax.set(xlim=lim, ylim=lim)

animations = []
for ax, trajectories, title in zip(axs, [sgld_trajectories, svgd_trajectories], ["SGLD", "SVGD"]):
    ax.set_title(title)
    plot.plot_fun_2d(target.pdf, lims=lim, ax=ax)
    animations.append(plot.animate_array(trajectories, fig, ax, interval=3))


get_ipython().run_line_magic("matplotlib", " inline")
fig, axs = plt.subplots(1, 3, figsize=[20, 6])
axs = axs.flatten()
lims=(-2, 2)
for ax in axs: ax.set(xlim=lims, ylim=lims)
particles.plot_final(target, ax=axs[0])
svgd_particles.plot_final(target, ax=axs[1])
plot.scatter(target.sample(60), ax=axs[2])







