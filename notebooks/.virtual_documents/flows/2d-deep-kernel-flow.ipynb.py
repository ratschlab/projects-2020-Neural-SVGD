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


get_ipython().run_line_magic("matplotlib", " inline")
# setup = distributions.ring_target
# target, proposal = setup.get()
target = distributions.Banana([0, 0], [4, 1])
proposal = distributions.Gaussian([-5, -5], 1)
setup = distributions.Setup(target, proposal)
setup.plot(lims=(-15, 15))


from flows import default_noise_level, default_num_particles, default_num_steps, default_patience
def deep_kernel_flow(key,
                     setup,
                     n_particles=default_num_particles,
                     n_steps=default_num_steps,
                     sizes=[32, 32, 1],
                     particle_lr=1e-1,
                     learner_lr=1e-2,
                     lambda_reg=1/2,
                     noise_level=default_noise_level):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()
    learner = models.KernelLearner(target=target,
                                   key=keya,
                                   sizes=sizes,
                                   learning_rate=learner_lr,
                                   lambda_reg=lambda_reg,
                                   patience=default_patience)
    
    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 proposal=proposal,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr)

    for _ in tqdm(range(n_steps)):
        try:
            key, subkey = random.split(key)
            learner.train(particles.next_batch, key=subkey, n_steps=500, noise_level=noise_level)
            particles.step(learner.get_params(), noise_pre=noise_level)
        except FloatingPointError as err:
            warnings.warn(f"Caught floating point error")
            return learner, particles, err
    return learner, particles, None


get_ipython().run_line_magic("autoreload", "")


n_steps = 50
noise = 1.
key, subkey = random.split(key)
learner, neural_particles, err = deep_kernel_flow(subkey, setup, n_steps=n_steps, sizes=[32, 32, 2], particle_lr=1e-2, noise_level=noise)
# kernel_gradient, kernel_particles, err = flows.score_flow(subkey, setup, n_steps=n_steps, particle_lr=1e-2)


get_ipython().run_line_magic("matplotlib", " inline")
n_steps = neural_learner.rundata["train_steps"]
plt.plot(n_steps)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(neural_learner.rundata["training_loss"], "--.", label="Trainging Loss")
plt.plot(neural_learner.rundata["validation_loss"], "--.", label="Validation Loss")
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
