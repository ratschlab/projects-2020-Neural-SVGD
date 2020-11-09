get_ipython().run_line_magic("load_ext", " autoreload")
from jax import config
config.update("jax_debug_nans", False)
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


def plot_samples(target_pdf, sample_list, lims=(-4, 4)):
    """sample_list: iterable of sets of samples"""
    fig, axs = plt.subplots(1, 3, figsize=[30, 9])
    titles = ("Neural SVGD", "SVGD", "True samples")
    for ax, samples, title in zip(axs.flatten(), sample_list, titles):
        ax.set(xlim=lims, ylim=lims)
        plot.plot_fun_2d(target_pdf, lims=lims, ax=ax, alpha=0.5)
#         plot.plot_gradient_field(utils.negative(neural_learner.grads), ax, lims=lims)
        plot.scatter(samples, ax=ax)
        ax.set_title(title)


def animate(target_pdf, traj_list, lims=(-4, 4), interval=100):
    """Remember to activate get_ipython().run_line_magic("matplotlib", " widget\"\"\"")
    n = len(traj_list)
    fig, axs = plt.subplots(1, n, figsize=[6*n, 5])
    titles = ("Neural SVGD", "SVGD")
    anims = []
    for ax, traj, title in zip(axs.flatten(), traj_list, titles):
        ax.set(xlim=lims, ylim=lims)
        plot.plot_fun_2d(target.pdf, lims=lims, ax=ax, alpha=0.5)
        anims.append(plot.animate_array(traj, fig, ax, interval=interval))
    return anims, fig, axs


neural_trajectories = []
svgd_trajectories = []


get_ipython().run_line_magic("autoreload", "")


setup = distributions.squiggle_target
target, proposal = setup.get()


n_steps = 200
n_particles = 100

particle_lr = 1e-2
learner_lr = 2e-3

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list)


get_ipython().run_line_magic("matplotlib", " inline")
plt.plot(neural_learner.rundata["train_steps"])


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[15, 5])
for loss, label in [(neural_learner.rundata[k], k) for k in ("training_loss", "validation_loss")]:
    plt.plot(loss, "--o", label=label)
    
plt.legend()


neural_trajectories = []
svgd_trajectories = []


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)

neural_trajectories.append(traj)
svgd_trajectories.append(traj_svgd)


testt = np.array([p.test for p in neural_particles.rundata["particles"]])


# i = 0
# traj, traj_svgd = [lst[i] for lst in (neural_trajectories, svgd_trajectories)]
# get_ipython().run_line_magic("matplotlib", " widget")
# lims=(-4, 4)
# ans, fig, axs = animate(target.pdf, [traj, traj_svgd], lims=lims, interval=100)
# # ans.append(plot.animate_array(testt, fig, axs[0]))
# ans


setup = distributions.mix_of_gauss
target, proposal = setup.get()


n_steps = 500
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
mix_lims = (-2, 2)
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list, mix_lims)


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)

neural_trajectories.append(traj)
svgd_trajectories.append(traj_svgd)


# i = 1
# traj, traj_svgd = [lst[i] for lst in (neural_trajectories, svgd_trajectories)]
# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=mix_lims, interval=10)


target, proposal = distributions.mix_of_gauss.get()
proposal = distributions.Gaussian([0, 0], [1, 1])
setup = distributions.Setup(target, proposal)


n_steps = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list, mix_lims)


setup = distributions.ring_target
target, proposal = setup.get()


n_steps = 300
# n_particles = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)

neural_trajectories.append(traj)
svgd_trajectories.append(traj_svgd)


sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list, (-15, 15))


# i = 2
# traj, traj_svgd = [lst[i] for lst in (neural_trajectories, svgd_trajectories)]
# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=(-15, 15), interval=10)


setup = distributions.funnel
target, proposal = setup.get()


n_steps = 500
# n_particles = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
funnel_lims=(-15, 15)
plot_samples(target.pdf, sample_list, funnel_lims)


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)

neural_trajectories.append(traj)
svgd_trajectories.append(traj_svgd)


# i = 3
# traj, traj_svgd = [lst[i] for lst in (neural_trajectories, svgd_trajectories)]
# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=funnel_lims, interval=10)


setup = distributions.banana_target
target, proposal = setup.get()


n_steps = 800
# n_particles = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
# sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
banana_lims=(-10, 10)
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list, banana_lims)


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)

neural_trajectories.append(traj)
svgd_trajectories.append(traj_svgd)


# i = 4
# traj, traj_svgd = [lst[i] for lst in (neural_trajectories, svgd_trajectories)]
# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=banana_lims, interval=10)


# get_ipython().run_line_magic("matplotlib", " widget")
plt.plot(neural_learner.rundata["training_loss"])
plt.plot(neural_learner.rundata["validation_loss"])



