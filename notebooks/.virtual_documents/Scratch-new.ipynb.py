get_ipython().run_line_magic("load_ext", " autoreload")

import sys
import copy
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning")
import json
import collections
import itertools
from functools import partial
import importlib

import numpy as onp
from jax.config import config
config.update("jax_debug_nans", False)
# config.update("jax_log_compiles", True)
# config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import optax
import matplotlib.pyplot as plt

import numpy as onp
import jax
import pandas as pd
import haiku as hk
import ot


import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import kernel_learning
import models
import flows

from jax.experimental import optimizers

key = random.PRNGKey(0)
key, subkey = random.split(key)

from jax.scipy.stats import norm

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


setup = distributions.funnel
target, proposal = setup.get()
lr = 1e-3
n_steps = 500

kernel = tfp.mcmc.UncalibratedLangevin(target_log_prob_fn=target.logpdf, step_size=lr)

@jit
def run_chain(key, state):
    return tfp.mcmc.sample_chain(n_steps,
      current_state=state,
      kernel=kernel,
      trace_fn = None,
      num_burnin_steps=0,
      seed=key)


key, subkey = random.split(key)
state = proposal.sample(100, subkey)[0]
p1 = run_chain(subkey, state)

key, subkey = random.split(key)
grad, particles, err = flows.sgld_flow(subkey, setup, n_particles=1, n_steps=n_steps, particle_lr=lr)
p2 = np.squeeze(particles.rundata["particles"].training)


fig, ax = plt.subplots(figsize=[13, 8])
ax.set(xlim=(-15, 15), ylim=(-15, 15))
plot.plot_fun_2d(target.pdf, lims=(-15,15))

key, subkey = random.split(key)
plt.scatter(*np.rollaxis(target.sample(100, subkey), 1), alpha=0.2)

plot.scatter(p1, alpha=0.1, label="TFP")
plot.scatter(p2, alpha=0.1, label="my implementation")
plt.legend()


n_particles = 100
key, subkey = random.split(key)
grad, particles, err = flows.sgld_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, particle_lr=lr)

key, subkey = random.split(key)
state = proposal.sample(n_particles, subkey)
pp1 = vmap(run_chain)(random.split(subkey, n_particles), state)
pp2 = particles.rundata["particles"].training


fig, ax = plt.subplots(figsize=[13, 8])
ax.set(xlim=(-15, 15), ylim=(-15, 15))
plot.plot_fun_2d(target.pdf, lims=(-15,15))

key, subkey = random.split(key)
plt.scatter(*np.rollaxis(target.sample(100, subkey), 1), alpha=0.2)

plot.scatter(pp1[:, -1, :], alpha=0.8, label="TFP")
plot.scatter(pp2[-1], alpha=0.8, label="my implementation")
plt.legend()


get_ipython().run_line_magic("autoreload", "")


# opt = utils.polynomial_sgld(1.)
opt = utils.scaled_sgld(subkey, 1., utils.polynomial_schedule)


params = np.array(100.)
state = opt.init(params)
grads = np.array(0.)

p = []
u = []
for _ in range(1000):
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    p.append(params)
    u.append(updates)


plt.plot(p)


plt.subplots(figsize=[15, 8])
plt.plot(u)
plt.plot([np.sqrt(2*utils.polynomial_schedule(k)) for k in range(1000)])


np.mean(np.array(u))


np.var(np.array(u))



