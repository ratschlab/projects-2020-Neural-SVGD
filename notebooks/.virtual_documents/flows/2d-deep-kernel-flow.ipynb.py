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


d = 50
target = distributions.Gaussian(np.zeros(d), np.ones(d))
proposal = distributions.Gaussian(np.zeros(d), np.ones(d)*25)
setup = distributions.Setup(target, proposal)


# get_ipython().run_line_magic("matplotlib", " inline")
# # setup = distributions.banana_target
# # target, proposal = setup.get()
# target = distributions.Gaussian([0, 0], [1e-2, 1])
# proposal = distributions.Gaussian([0,0], 1)
# setup = distributions.Setup(target, proposal)
# # target = distributions.Banana([0, 0], [4, 1])
# # proposal = distributions.Gaussian([-5, -5], 1)
# # setup = distributions.Setup(target, proposal)
# setup.plot(lims=(-15, 15))


get_ipython().run_line_magic("autoreload", "")


key, subkey = random.split(key)


n_steps = 400
noise = 0.
n_particles = 50
sizes = None
# key, subkey = random.split(key)

learner, neural_particles, err1         = flows.deep_kernel_flow(subkey, setup, n_steps=n_steps, particle_lr=1e-2, noise_level=noise, sizes=sizes, learner_lr=1e-1)
# kernel_gradient, kernel_particles, err2 = flows.svgd_flow(subkey, setup, n_steps=n_steps, particle_lr=1e-2, noise_level=0, scaled=True)


plt.subplots(figsize=[8, 6])
plt.ylim((0, 5))
stdneur = np.array(neural_particles.rundata["training_std"])
stdsvgd = kernel_particles.rundata["training_std"]
plt.plot(np.mean(stdneur, axis=1), color="tab:blue", label="Learned Kernel")
plt.plot(onp.mean(stdsvgd, axis=1), color="tab:orange", label="SVGD")
_ = plt.axhline(y=1, color="green", linestyle="--")
plt.ylabel("Mean variance")
plt.xlabel("Step")
plt.legend()


stdneur = neural_particles.rundata["validation_std"]
stdsvgd = kernel_particles.rundata["validation_std"]
plt.plot(stdneur, color="tab:blue", label="neur")
_ = plt.plot(stdsvgd, color="tab:orange")
_ = plt.axhline(y=1, color="green", linestyle="--")
# plt.legend()


paramtree = utils.dict_concatenate(learner.rundata["model_params"])

leaves, structure = jax.tree_flatten(paramtree)

h = np.array(leaves[0])
bw = np.exp(h)
s = np.array(leaves[1])

get_ipython().run_line_magic("matplotlib", " inline")
plt.plot(h, "--", label="bandwidth", color="tab:blue")
# plt.plot(s, "--", label="scale", color="tab:orange")
plt.ylim((None, None))
# plt.legend()


learner.get_params()


## get_ipython().run_line_magic("matplotlib", " inline")
n_steps = learner.rundata["train_steps"]
plt.plot(n_steps)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(learner.rundata["training_loss"], "--.", label="Trainging Loss")
plt.plot(learner.rundata["validation_loss"], "--.", label="Validation Loss")
plt.legend()
# plt.ylim((-.2, 0))


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(learner.rundata["training_ksd"], "--.", label="Trainging KSD")
plt.plot(learner.rundata["validation_ksd"], "--.", label="Validation KSD")
plt.legend()
plt.yscale("log")


x = np.array([0, 0])
x50d = np.append(x, train_x[0, 2:])


x50d.shape


kernel = learner.get_kernel_fn(train_x)
def kfn(x):
    return kernel(train_x[0], x)
slice_dims=[0, 1]

def kfn_sliced(x):
    """2d slice of kfn"""
    x50d = np.append(x, train_x[0, 2:])
    return kfn(x50d)


kernel(train_x[0], train_x[0])


kfn_sliced(train_x[0, :2]-.01)


train_x, val_x, t_x = neural_particles.get_params(split_by_group=True)
final_svgd, *_ = kernel_particles.get_params(split_by_group=True)

fig, axs = plt.subplots(1, 3, figsize=[25, 8])
axs = axs.flatten()
lim=(-10, 10)
for ax in axs:
    ax.set(xlim=lim, ylim=lim)

plot.plot_fun_2d(distributions.Gaussian([0,0], [1,1]).pdf, ax=axs[0], lims=lim)
for particles in (train_x, val_x):
    axs[0].scatter(particles[:, 20], particles[:, 11])
axs[0].set_title("Neural SVGD") 


trajectory.shape


trajectory = np.array(neural_particles.rundata["particles"])
trajectory_projected = trajectory[:, :, [0, 1]]
t_train, t_val, t_t = [trajectory_projected[:, idx, :] for idx in neural_particles.group_idx]


get_ipython().run_line_magic("matplotlib", " widget")
fig, ax = plt.subplots(figsize=[8, 8])
interval=10
anim=[]
for t in [t_train, t_val]:
    anim.append(plot.animate_array(t, fig=fig, ax=ax, interval=interval))


## get_ipython().run_line_magic("matplotlib", " inline")
fig, axs = plt.subplots(1, 3, figsize=[25, 8])
axs = axs.flatten()
lim=(-10, 10)
for ax in axs:
    ax.set(xlim=lim, ylim=lim)
ax1 = neural_particles.plot_final(ax=axs[0], target=target, cmap="Greens")
ax1.set_title("Neural SVGD")

ax2 = kernel_particles.plot_final(ax=axs[1], target=target, cmap="Greens")
ax2.set_title("SVGD")

ax = axs[2]
plot.plot_fun_2d(target.pdf, lims=lim, ax=ax, cmap="Greens")
plot.scatter(target.sample(50), ax=ax, color="tab:orange")
ax.set_title("True Samples")


neural_trajectories = np.asarray(neural_particles.rundata["particles"])
kernel_trajectories = np.asarray(kernel_particles.rundata["particles"])


np.var(neural_trajectories[-1], axis=0)


np.var(kernel_trajectories[-1], axis=0)


get_ipython().run_line_magic("matplotlib", " widget")
fig, axs = plt.subplots(1, 2, figsize=[14, 6])
axs=axs.flatten()
lim=(-10, 10)
for ax in axs:
    ax.set(xlim=lim, ylim=lim)

animations = []
for ax, trajectories, title in zip(axs, [neural_trajectories, kernel_trajectories], ["Neural", "Kernel"]):
    ax.set_title(title)
    plot.plot_fun_2d(target.pdf, lims=lim, ax=ax)
    animations.append(plot.animate_array(trajectories, fig, ax))






