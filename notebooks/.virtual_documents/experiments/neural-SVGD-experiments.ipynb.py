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
    
key = random.PRNGKey(0)


get_ipython().run_line_magic("autoreload", "")


d = 50
variances = np.logspace(-2, 0, num=d)
target = distributions.Gaussian(np.zeros(d), variances)
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
setup = distributions.Setup(target, proposal)


n_steps = 3500
particle_lr = 1e-2
learner_lr = 1e-2
n_particles = 1000
key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, d], patience=0, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=100,         n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=5)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


raise err1


# get final particles
target_cov = target.cov
final_particles_list = []
for p in [neural_particles, svgd_particles, sgld_particles]:
    particles = p.get_params()
    final_particles_list.append(particles.training)
covs = [target.cov] + [np.cov(p, rowvar=False) for p in final_particles_list]
titles = ["True", "neural", "svgd", "sgld"]
neural_final, svgd_final, sgld_final = final_particles_list


fig, axs = plt.subplots(1, 4, figsize=[18, 5])
axs = axs.flatten()
for ax, cov, title in zip(axs, covs, titles):
    ax.imshow(cov)
    ax.set_title(title)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 7])
num=3
grid = np.arange(0, d*num, step=num)

for cov_diagonal, name, i in zip([np.diag(cov) for cov in covs[1:]], ["Neural", "SVGD", "SGLD"], range(3)):
    plt.bar(grid+i, cov_diagonal, alpha=0.5, label=name)

plt.scatter(grid, np.diag(target.cov), marker="_", label="True")
plt.legend()
plt.yscale("log")
plt.ylim((1e-3, 1))


# get_ipython().run_line_magic("matplotlib", " inline")
# fig, ax = plt.subplots(figsize=[10, 8])
# # plt.ylim((0, 5))
# stdneur = np.array(neural_particles.rundata["training_std"])
# # stdsvgd = svgd_particles.rundata["training_std"]
# # stdsgld = sgld_particles.rundata["training_std"]


# # plt.plot(onp.mean(neural_particles.rundata["validation_std"], axis=1), color="blue", label="Neural Score Matching: Validation particles")
# plt.plot(stdneur, color="tab:blue", label="Neural SVGD")
# # plt.plot(stdsvgd, color="tab:orange", label="SVGD")
# # plt.plot(stdsgld, color="tab:green", label="SGLD")

# [plt.axhline(y=y, color="black", linestyle="--") for y in np.sqrt(variances)]

# plt.ylabel("Standard Deviation")
# plt.xlabel("Step")

# plt.legend()
# plt.yscale("log")


neural_trajectory = np.array(neural_particles.rundata["particles"])
slice_idx = [0, 1]
neural_trajectory_sliced = neural_trajectory[:, :, slice_idx]
sliced_target = distributions.Gaussian([0, 0], variances[slice_idx])

# get_ipython().run_line_magic("matplotlib", " widget")
# fig, ax = plt.subplots(figsize=[10, 10])
# lim=(-11, 11)
# ax.set(xlim=lim, ylim=lim)

# # plot.plot_fun_2d(sliced_target.pdf, lims=lim, ax=ax, alpha=0.5)
# plot.animate_array(neural_trajectory_sliced, fig, ax, interval=100)


# sample m points from the corners of d-simplex
d = 3
m = 3 # =< d
idx = random.choice(key, d, (m,), replace=False)
means = np.eye(d)[idx] * 5

target = distributions.GaussianMixture(means, 1, np.ones(m))
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
setup = distributions.Setup(target, proposal)


get_ipython().run_line_magic("autoreload", "")


n_steps = 500
noise = 0.
particle_lr = 1e-2
learner_lr = 1e-2
n_particles = 1000
key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=noise, sizes=[32, 32, d], patience=0, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=100, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


# get final particles
target_cov = target.cov
final_particles_list = []
for p in [neural_particles, svgd_particles, sgld_particles]:
    particles = p.get_params()
    final_particles_list.append(particles.training)
covs = [target.cov] + [np.cov(p, rowvar=False) for p in final_particles_list]
titles = ["True", "neural", "svgd", "sgld"]
neural_final, svgd_final, sgld_final = final_particles_list

true_samples = target.sample(1000)


get_ipython().run_line_magic("matplotlib", " inline")
fig = plt.figure(figsize=[13, 13])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*np.rollaxis(neural_final, 1))
ax.scatter(*np.rollaxis(sgld_final, 1))
ax.scatter(*np.rollaxis(target.means, 1), s=4000, color="red")


neural_trajectory = np.array(neural_particles.rundata["particles"])
sgld_trajectory = np.array(sgld_particles.rundata["particles"])

neural_trajectory_sliced = neural_trajectory[:, :200, slice_idx]


get_ipython().run_line_magic("matplotlib", " inline")
fig, ax = plt.subplots(figsize=[10, 10])
_ = plt.plot(sgld_trajectory[:, :, 0])


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(neural_learner.rundata["training_loss"], "--.", label="Trainging Loss")
plt.plot(neural_learner.rundata["validation_loss"], "--.", label="Validation Loss")
# plt.plot(neural_learner.rundata["test_loss"], "--.", label="Test Loss")
# plt.ylim(0., 1e35)
plt.legend()
plt.yscale("log")


# fig, axs = plt.subplots(1, 4, figsize=[18, 5])
# axs = axs.flatten()
# for ax, cov, title in zip(axs, covs, titles):
#     ax.imshow(undo_rot(cov))
#     ax.set_title(title)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 7])
num=3
grid = np.arange(0, d*num, step=num)

for cov_diagonal, name, i in zip([np.diag(undo_rot(cov)) for cov in covs[1:]], ["Neural", "SVGD", "SGLD"], range(3)):
    plt.bar(grid+i, cov_diagonal, alpha=0.5, label=name)

plt.scatter(grid, np.diag(undo_rot(target.cov)), marker="_", label="True")
plt.legend()
plt.yscale("log")


fig, axs = plt.subplots(1, 4, figsize=[18, 5])
axs = axs.flatten()
for ax, cov, title in zip(axs, covs, titles):
    ax.imshow(cov)
    ax.set_title(title)


# get_ipython().run_line_magic("matplotlib", " inline")
# fig, ax = plt.subplots(figsize=[15, 25])
# # plt.ylim((0, 5))
# stdneur = np.array(neural_particles.rundata["training_std"])
# # stdsvgd = svgd_particles.rundata["training_std"]
# # stdsgld = sgld_particles.rundata["training_std"]


# # plt.plot(onp.mean(neural_particles.rundata["validation_std"], axis=1), color="blue", label="Neural Score Matching: Validation particles")
# plt.plot(stdneur, color="tab:blue", label="Neural Score Matching")
# # plt.plot(stdsvgd, color="tab:orange", label="SVGD")
# # plt.plot(stdsgld, color="tab:green", label="SGLD")

# [plt.axhline(y=y, color="black", linestyle="--") for y in np.sqrt(variances)]

# plt.ylabel("Standard Deviation")
# plt.xlabel("Step")

# plt.legend()
# # plt.yscale("log")


slice_idx = [0, 9]


particles, _ = neural_particles.get_params(split_by_group=True)
# _, particles = neural_particles.get_params(split_by_group=True)
# x0 = particles[0]
x0 = np.zeros(d)

def sliced_grads(xs):
    """function (n, 2) to (n, 2)"""
    n = len(xs)
    x0s = np.tile(x0, (n, 1)) # shape (n, d)
    xs_nd = index_update(x0s, index[:, slice_idx], xs)
    vec_nd = neural_learner.grads(xs_nd)
    return vec_nd[:, slice_idx]

sliced_target = distributions.Gaussian([0, 0], variances[slice_idx])
approx_q = distributions.Gaussian(np.mean(particles, axis=0)[slice_idx], np.var(particles, axis=0)[slice_idx])
sliced_grads_true = vmap(grad(lambda x: sliced_target.logpdf(x) - approx_q.logpdf(x)))

sliced_particles = particles[:, slice_idx] 


get_ipython().run_line_magic("matplotlib", " inline")
fig, axs = plt.subplots(1, 2, figsize=[22, 10])
axs = axs.flatten()
# lims=(-1/4, 1/4)
lims=(-11, 11)
# lims = utils.get_particle_lims(particles[:, slice_idx])
true_samples = target.sample(1000)
for ax, grad_fn in zip(axs, [sliced_grads, sliced_grads_true]):
    ax.set(xlim=lims, ylim=lims)
#     plot.plot_fun_2d(sliced_target.pdf, lims=lims, ax=ax, alpha=0.5)
    plot.plot_gradient_field(grad_fn, ax, lims=lims)
    plot.scatter(sliced_particles, ax=ax)
    plot.scatter(true_samples[:, slice_idx], ax=ax)


get_ipython().run_line_magic("matplotlib", " inline")
# fig, ax = plt.subplots(figsize=[10, 10])
# _ = plt.plot(neural_trajectory[:, :, 9])


neural_trajectory_sliced.shape


np.var(neural_trajectory[800], axis=0)[slice_idx]


get_ipython().run_line_magic("matplotlib", " widget")
fig, ax = plt.subplots(figsize=[10, 10])
lim=(-11, 11)
# lim=utils.get_particle_lims(sliced_particles)
ax.set(xlim=lim, ylim=lim)
# plot.plot_fun_2d(sliced_target.pdf, lims=lim, ax=ax, alpha=0.5)
plot.animate_array(neural_trajectory_sliced, fig, ax, interval=100)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[20, 6])
plt.plot(neural_learner.rundata["training_loss"], "--.", label="Trainging Loss")
plt.plot(neural_learner.rundata["validation_loss"], "--.", label="Validation Loss")
# plt.plot(neural_learner.rundata["test_loss"], "--.", label="Test Loss")
# plt.ylim(-100, 100)
plt.legend()


psvgd, *_ = svgd_particles.get_params(split_by_group=True)
metrics.compute_final_metrics(psvgd, target)


p, *_ = neural_particles.get_params(split_by_group=True)
metrics.compute_final_metrics(p[:100], target)


p, *_ = sgld_particles.get_params(split_by_group=True)
metrics.compute_final_metrics(p[:100], target)



