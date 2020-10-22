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
import optax

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
    return anims


class NeuralSampler(hk.Module):
    def __init__(self, layer_sizes: list = [2, 512, 2], batch_size=100, name: str = None):
        super().__init__(name=name)
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

    def __call__(self):
        mlp = hk.nets.MLP(output_sizes=self.layer_sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        key = hk.next_rng_key()
        input_shape = (self.batch_size, 2) if self.batch_size else (2,)
        xi = random.normal(key, shape=input_shape)
        return mlp(xi)

sampler = hk.transform(lambda: NeuralSampler(batch_size=1000)())
key, subkey = random.split(key)
sampler_params = sampler.init(key)

sampler_opt = optax.adam(0.01)
state = sampler_opt.init(sampler_params)

plot.scatter(sampler.apply(sampler_params, key))
plot.scatter(random.normal(subkey, (100,2)))
plt.xlim((-3,3))
plt.ylim((-3,3))


setup = distributions.squiggle_target
target, proposal = setup.get()

learner = models.SDLearner(target.logpdf, target.d)


def sampledot(sampler_params, key, v):
    """v needs to have shape (m, n), where m is the nr of samples and n the sample dimension"""
    return np.mean(vmap(np.dot)(sampler.apply(sampler_params, key), v))

@jit
def sampler_grads(key, sampler_params, gradient_params):
    """Compute sample gradient and backpropagate into sampler"""
    dkl = learner.gradient(gradient_params, None, sampler.apply(sampler_params, key))
    return grad(sampledot)(sampler_params, key, dkl)


def gradient_train(sampler_params, key, n_steps=50):
    keys = random.split(key, n_steps)
    for key in keys:
        samples = sampler.apply(sampler_params, key)
        learner.step(samples, samples)


for _ in range(2500):
    key, subkey = random.split(key)
    gradient_train(sampler_params, subkey, n_steps=10)
    
    key, subkey = random.split(key)
    grads = sampler_grads(subkey, sampler_params, learner.get_params())
    grads, state = sampler_opt.update(grads, state, sampler_params)
    sampler_params = optax.apply_updates(sampler_params, grads)


key, subkey = random.split(key)
samples = sampler.apply(sampler_params, subkey)
fig, ax = plt.subplots(figsize=[9,9])
lims=(-4, 4)
ax.set(ylim=lims, xlim=lims)
plot.plot_fun_2d(target.pdf, lims=lims, num_gridpoints=300)
plot.scatter(samples)


key, subkey = random.split(key)
samples = target.sample(1000)
fig, ax = plt.subplots(figsize=[9,9])
lims=(-4, 4)
ax.set(ylim=lims, xlim=lims)
plot.plot_fun_2d(target.pdf, lims=lims, num_gridpoints=300)
plot.scatter(samples)


plt.plot(learner.rundata["training_loss"])
# plt.plot(learner.rundata["validation_loss"])


plt.plot(learner.rundata["training_sd"])
# plt.plot(learner.rundata["validation_loss"])


get_ipython().run_line_magic("autoreload", "")


setup = distributions.squiggle_target
target, proposal = setup.get()


n_steps = 200
n_particles = 100

particle_lr = 1e-2
learner_lr = 2e-3

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, 2], patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list)


get_ipython().run_line_magic("matplotlib", " inline")
plt.subplots(figsize=[15, 5])
for loss in [neural_learner.rundata[k] for k in ("training_loss", "validation_loss")]:
    plt.plot(loss)


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)


# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=lims, interval=10)


setup = distributions.mix_of_gauss
target, proposal = setup.get()


n_steps = 500
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, 2], patience=10, learner_lr=learner_lr)
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


# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=mix_lims, interval=10)


target, proposal = distributions.mix_of_gauss.get()
proposal = distributions.Gaussian([0, 0], [1, 1])
setup = distributions.Setup(target, proposal)


n_steps = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, 2], patience=10, learner_lr=learner_lr)
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
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, 2], patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


# sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
# plot_samples(target.pdf, sample_list, (-15, 15))


setup = distributions.funnel
target, proposal = setup.get()


n_steps = 500
# n_particles = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, 2], patience=10, learner_lr=learner_lr)
svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)


get_ipython().run_line_magic("matplotlib", " inline")
sample_list = [p.particles.training for p in (neural_particles, svgd_particles)] + [target.sample(n_particles)]
plot_samples(target.pdf, sample_list, (-15, 15))


traint = [p.training for p in neural_particles.rundata["particles"]]
traint_svgd = [p.training for p in svgd_particles.rundata["particles"]]

traj = np.array(traint)
traj_svgd = np.array(traint_svgd)


# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=lims, interval=10)


setup = distributions.banana_target
target, proposal = setup.get()


n_steps = 800
# n_particles = 300
# particle_lr = 1e-2
# learner_lr = 1e-1

key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., sizes=[32, 32, 2], patience=10, learner_lr=learner_lr)
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


# get_ipython().run_line_magic("matplotlib", " widget")
# animate(target.pdf, [traj, traj_svgd], lims=banana_lims, interval=10)



