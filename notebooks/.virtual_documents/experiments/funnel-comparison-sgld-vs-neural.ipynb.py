get_ipython().run_line_magic("load_ext", " autoreload")
from jax import config
config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
from matplotlib.animation import FuncAnimation

import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax.ops import index_update, index
import matplotlib.pyplot as plt
import matplotlib
import numpy as onp
import jax
import pandas as pd

import utils
import plot
import distributions
import models
import flows
from tqdm import tqdm
key = random.PRNGKey(0)

import kernels
import metrics
# import seaborn as sns
# sns.set_theme()


get_ipython().run_line_magic("autoreload", "")


# set up exporting
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.unicode_minus': False, # avoid unicode error on saving plots with negative numbers (??)
})

figure_path = "/home/lauro/documents/msc-thesis/thesis/figures/"
# save figures by using plt.savefig('path/to/fig')
# remember that latex textwidth is 5.4in
# so use figsize=[5.4, 4], for example
printsize = [5.4, 4]


n_steps = 5000
particle_lr = 1e-2
learner_lr = 1e-4
n_particles = 200
d = 2
PATIENCE = 0
# PATIENCE = 15 # try this


target = distributions.Funnel(d)
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
funnel_setup = distributions.Setup(target, proposal)
target_samples = target.sample(n_particles)


true_samples = target.sample(n_particles)
def plot_true(idx=np.array([0, -1]), ax=None):
    if ax is None:
        ax = plt.gca()
    lims=(-15, 15)
    ax.set(xlim=lims, ylim=lims)
    ax.scatter(*np.rollaxis(true_samples[:, idx], 1), alpha=0.25, label="True", marker=".")


key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, funnel_setup, n_particles=n_particles, n_steps=5000, particle_lr=particle_lr, patience=PATIENCE, learner_lr=learner_lr, aux=False, compute_metrics=metrics.get_funnel_tracer(target_samples))
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=5000, particle_lr=particle_lr, compute_metrics=metrics.get_funnel_tracer(target_samples))
sgld_gradient2, sgld_particles2, err4    = flows.sgld_flow(     subkey, funnel_setup, n_particles=n_particles, n_steps=5000, particle_lr=particle_lr/5, compute_metrics=metrics.get_funnel_tracer(target_samples))
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=5000, particle_lr=particle_lr*5, scaled=True,  bandwidth=None, compute_metrics=metrics.get_funnel_tracer(target_samples))

# Note: I scaled the svgd step-size (by hand) so that it is maximial while still converging to a low MMD.


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[15, 8])

typ = "funnel_"

plt.plot(*zip(*neural_particles.rundata[typ+"mmd"]), label="Neural gradient flow")
plt.plot(*zip(*sgld_particles.rundata[typ+"mmd"]), label="Langevin")
plt.plot(*zip(*sgld_particles2.rundata[typ+"mmd"]), label="Langevin (reduced step-size)")
plt.plot(*zip(*svgd_particles.rundata[typ+"mmd"]), label="SVGD")


plt.xlabel("Iteration")
plt.ylabel("MMD(samples, target)")

plt.legend()


# svgd scatterplot
fig, ax = plt.subplots(figsize=[5,5])
plot_true(ax=ax)
ax.scatter(*np.rollaxis(svgd_particles.particles.training, 1), alpha=0.55)
ax.legend()
ax.set_title("SVGD")
# ax.set(xlim=lims, ylim=lims)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=printsize)

typ = "funnel_"

plt.plot(*zip(*neural_particles.rundata[typ+"mmd"]), label="Neural gradient flow")
plt.plot(*zip(*sgld_particles.rundata[typ+"mmd"]), label="Langevin")
plt.plot(*zip(*sgld_particles2.rundata[typ+"mmd"]), label="Langevin (reduced step-size)")
plt.plot(*zip(*svgd_particles.rundata[typ+"mmd"]), label="SVGD")
plt.legend()

plt.xlabel("Iteration")
plt.ylabel("MMD(samples, target)")

# plt.savefig(figure_path + "funnel_mmd.pgf")


def plot_projection(idx, figsize=[20, 5]):
    sample_list = [p.particles.training for p in (neural_particles, sgld_particles, sgld_particles2)]
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    titles = ("Neural gradient flow", "Langevin") #+ ("Langevin (smaller stepsize)",)
    for samples, ax, title in zip(sample_list, axs.flatten(), titles):
        plot_true(idx, ax)
        ax.scatter(*np.rollaxis(samples[:, idx], 1), alpha=0.5, marker=".")
        ax.legend()
        ax.set_title(title)
        ax.set(xlim=lims, ylim=lims)


get_ipython().run_line_magic("matplotlib", " inline")
lims=(-15, 15)

idx = np.array([0, -1])
plot_projection(idx, figsize=[20, 8])


idx = np.array([0, -1])
plot_projection(idx, figsize=printsize)
# plt.savefig(figure_path + "funnel_scatter.pgf")


# fig, axs = plt.subplots(1, 2, figsize=[15, 7])
# for ax in axs:
#     plot.scatter(vmap(kernels.desvgd_particlesize)(true_samples), ax=ax)
#     ax.set(ylim=[-3, 3], xlim=[-3, 3])

# plot.scatter(vmap(kernels.defunnelize)(neural_particles.particles.training), ax = axs[0])
# plot.scatter(vmap(kernels.defunnelize)(sgld_particles.particles.training), ax = axs[1])

# mmd = jit(metrics.get_mmd(kernels.get_funnel_kernel(1.)))

# true_samples = target.sample(n_particles*10)

# mmd(*[samples for samples in (neural_particles.particles.training, true_samples)])

# mmd(*[samples for samples in (sgld_particles.particles.training, true_samples)])


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[15, 5])
plt.plot(neural_learner.rundata["training_loss"])
plt.plot(neural_learner.rundata["validation_loss"])


get_ipython().run_line_magic("matplotlib", " inline")
plt.plot(neural_learner.rundata["train_steps"])


idx = np.array([0, -1])


neural_particles.rundata.keys()


trajectory = neural_particles.rundata["particles"].training
trajectory_projected = trajectory[:, :, idx]



# get_ipython().run_line_magic("matplotlib", " widget")
# lims = (-15, 15)
# fig, axs = plt.subplots(1, 3, figsize=[25,8])
# for ax in axs:
#     ax.scatter(*np.rollaxis(target.sample(n_particles)[:, idx], 1), label="True", alpha=0.25)
#     ax.set(xlim=lims, ylim=lims)

    
# interval = 10
# a=[]
# a.append(plot.animate_array(trajectory_projected, fig, ax=axs[0], interval=interval))
# a.append(plot.animate_array(sgld_particles.rundata["particles"].training, ax=axs[1], interval=interval))
# # a.append(plot.animate_array(sgld_particles2.rundata["particles"].training, ax=axs[2], interval=interval))
# a


key, subkey = random.split(key)


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


d = 2
dfunnel = distributions.Funnel(d)
dproposal = distributions.Gaussian(np.ones(d), 1)
target_log_prob = dfunnel.logpdf

# kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(target_log_prob_fn=target_log_prob, step_size=1e-2)
# kernel = tfp.mcmc.NoUTurnSampler(target_log_prob, 1e-3)
kernel = tfp.mcmc.UncalibratedLangevin(target_log_prob_fn=target_log_prob, step_size=particle_lr/5)

@jit
def run_chain(key, state):
    return tfp.mcmc.sample_chain(1_000_000,
      current_state=state,
      kernel=kernel,
      trace_fn = None,
#       trace_fn=lambda _, results: results.target_log_prob,
      num_burnin_steps=0,
      seed=key)


key, subkey = random.split(key)
single_chain_init = dproposal.sample(1, subkey)[0]
key, subkey = random.split(key)
single_chain = run_chain(subkey, single_chain_init)


# del single_chain


spaced_samples = single_chain[np.arange(0, 1_000_000, 1000)]


mmd = metrics.get_mmd(kernel=kernels.get_funnel_kernel(1.))


sequential_mmd = mmd(target_samples, spaced_samples)


sequential_mmd


get_ipython().run_line_magic("matplotlib", " inline")
fig, ax = plt.subplots(figsize=printsize)
plot_true(ax=ax)
plot.scatter(spaced_samples, ax=ax, alpha=0.5, marker=".")
# plt.savefig(figure_path + "sequential_ula_funnel_scatter.pgf")


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=printsize)

typ = "funnel_"

plt.plot(*zip(*neural_particles.rundata[typ+"mmd"]), label="Neural gradient flow")
plt.plot(*zip(*sgld_particles.rundata[typ+"mmd"]), label="Langevin")
plt.plot(*zip(*sgld_particles2.rundata[typ+"mmd"]), label="Langevin (reduced step-size)")
plt.plot(*zip(*svgd_particles.rundata[typ+"mmd"]), label="SVGD")
plt.plot(*zip(*neural_particles.rundata[typ+"mmd"]), label="Neural gradient flow", color="tab:blue")

plt.axhline(y=sequential_mmd, label="Sequential Langevin", color="tab:green", linestyle="--")

plt.xlabel("Iteration")
plt.ylabel("MMD(samples, target)")

plt.legend()

# plt.savefig(figure_path + "funnel_mmd.pgf")


# get_ipython().run_line_magic("matplotlib", " widget")
# fig, ax = plt.subplots(figsize=[7,7])
# plot_true(ax=ax)
# plot.animate_array(single_chain[:, None, :], interval=1)


sdjkfdk


get_ipython().run_line_magic("autoreload", "")


particle_lr = 5e-3
learner_lr = 5e-5
n_particles = 200
d = 25
PATIENCE = 0
# PATIENCE = 15 # try this


target = distributions.Funnel(d)
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
funnel_setup = distributions.Setup(target, proposal)
target_samples = target.sample(500)


target_moment2 = np.array([9] + (d-1)*[90])


key, subkey = random.split(key)
neural_learner_d, neural_particles_d, err1 = flows.neural_svgd_flow(subkey, funnel_setup, n_particles=n_particles, n_steps=2000, particle_lr=particle_lr, patience=PATIENCE, learner_lr=learner_lr, aux=False, compute_metrics=metrics.get_2nd_moment_tracer(target_moment2))
sgld_gradient_d, sgld_particles_d, err3    = flows.sgld_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=2000, particle_lr=particle_lr, compute_metrics=metrics.get_2nd_moment_tracer(target_moment2))
# sgld_gradient2_d, sgld_particles2_d, err4    = flows.sgld_flow(     subkey, funnel_setup, n_particles=n_particles, n_steps=10000, particle_lr=particle_lr/5, compute_metrics=metrics.get_2nd_moment_tracer(target_moment2))


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[15, 8])

plt.plot(*zip(*neural_particles_d.rundata["second_error"]), label="Neural")
plt.plot(*zip(*sgld_particles_d.rundata["second_error"]), label="SGLD")
# plt.plot(*zip(*sgld_particles2_d.rundata["second_error"]), label="SGLD (reduced step-size)")
# plt.plot(*zip(*svgd_particles.rundata["second_error"]), label="SVGD")
# plt.yscale("symlog")

plt.xlabel("Iteration")
plt.ylabel("MMD(samples, target)")

plt.legend()


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=printsize)

typ = "funnel_"

plt.plot(*zip(*neural_particles_d.rundata[typ+"mmd"]), label="Neural gradient flow")
plt.plot(*zip(*sgld_particles_d.rundata[typ+"mmd"]), label="Langevin")
# plt.plot(*zip(*sgld_particles2_d.rundata[typ+"mmd"]), label="Langevin (reduced step-size)")
# plt.plot(*zip(*svgd_particles.rundata[typ+"mmd"]), label="SVGD")
plt.legend()

plt.xlabel("Iteration")
plt.ylabel("MMD(samples, target)")

# plt.savefig(figure_path + "funnel_mmd_3d.pgf")


sgld_particles_d.particles.training[:, idx].shape


np.rollaxis(sgld_particles_d.particles.training[:, idx], 1).shape


idx = np.array([0, -1])
plot_true(idx)
plt.scatter(*np.rollaxis(sgld_particles_d.particles.training[:, idx], 1), alpha=0.9)


def plot_projection(idx, figsize=[20, 5]):
    sample_list = [p.particles.training for p in (neural_particles_d, sgld_particles_d)]
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    titles = ("Neural gradient flow", "Langevin") #+ ("Langevin (smaller stepsize)",)
    for samples, ax, title in zip(sample_list, axs.flatten(), titles):
        plot_true(idx, ax)
        ax.scatter(*np.rollaxis(samples[:, idx], 1), alpha=0.9)
        ax.legend()
        ax.set_title(title)
        ax.set(xlim=lims, ylim=lims)


get_ipython().run_line_magic("matplotlib", " inline")
lims=(-15, 15)

idx = np.array([0, -1])
plot_projection(idx, figsize=[20, 8])


idx = np.array([0, -1])
plot_projection(idx, figsize=printsize)
# plt.savefig(figure_path + "funnel_scatter_3d.pgf")


idx = np.array([0, -1])


# get_ipython().run_line_magic("matplotlib", " widget")
# lims = (-15, 15)
# fig, axs = plt.subplots(1, 3, figsize=[25,8])
# for ax in axs:
#     plot_true(ax=ax)
    
# interval = 1
# a=[]
# a.append(plot.animate_array(neural_particles_d.rundata["particles"].training[:, :, idx], fig, ax=axs[0], interval=interval))
# a.append(plot.animate_array(  sgld_particles_d.rundata["particles"].training[:, :, idx], fig, ax=axs[1], interval=interval))
# # a.append(plot.animate_array(sgld_particles2.rundata["particles"].training, ax=axs[2], interval=interval))
# a









