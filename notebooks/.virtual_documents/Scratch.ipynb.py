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

import config

import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import kernel_learning

from jax.experimental import optimizers

key = random.PRNGKey(0)

from jax.scipy.stats import norm


def net_fn(batch):
    net = hk.nets.MLP([32, 32, 1], w_init=hk.initializers.VarianceScaling(2.0))
    return net(3*batch)
net = hk.transform(net_fn)

@jit
def loss_fn(params, batch):
    return np.mean((net.apply(params, None, batch) - np.sin(10*batch))**2)


# init net and optimizer
dummy_batch = onp.random.rand(10, 1)
params = net.init(key, dummy_batch)

opt = optax.sgd(1e-2)
state = opt.init(params)

losses=[]


for i in range(10**3):
    key, subkey = random.split(key)
    batch = random.uniform(subkey, shape=(100, 1), minval=-2, maxval=2)
    step_loss, grads = value_and_grad(loss_fn)(params, batch)
    grads, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, grads)
    losses.append(step_loss)
plt.plot(losses)
plt.yscale("log")


x = np.linspace(-1, 1, 100).reshape(100, 1)
plt.scatter(x, net.apply(params, None, x))


key, subkey = random.split(key)
net = nets.build_mlp([32, 32, 1], activate_final=True)
x_dummy = np.array([0.])
init_params = net.init(subkey, x_dummy)

@utils.reshape_input
def f(x):
    return net.apply(init_params, None, x)

xlims = (-100, 100)
plot.plot_fun(f, lims=xlims)


@jax.custom_transforms
def safe_sqrt(x):
    return np.sqrt(x)
jax.defjvp(safe_sqrt, lambda g, ans, x: 0.5 * g / np.where(x > 0, ans, np.inf) )


grad(lambda x: np.sqrt(x)**2)(0.)


grad(lambda x: safe_sqrt(x)**2)(0.) # baad


kernel = kernels.get_rbf_kernel(1)
q = distributions.Gaussian(0, 1)
s = q.sample(10_000)

@partial(jit, static_argnums=0)
def l2_and_rkhs_norm(kernel, x):
    """Return both norms of k(x, _)"""
    x = np.squeeze(np.array(x))
    def kx(y):
        return kernel(x, np.squeeze(y))
    l2_norm = utils.l2_norm(s, kx)
    rkhs_norm = kernel(x, x)
    return l2_norm, rkhs_norm


xgrid = np.linspace(-3, 3, num=50)
norms = vmap(l2_and_rkhs_norm, (None, 0))(kernel, xgrid)
norms = np.array(norms)


plt.plot(norms.T)


l2_and_rkhs_norm(kernel, 2.)


class Benchmarks():
    def __init__(target, proposal, lambda_reg):
        pass


def get_optimal_sd(key, lambda_reg, target, proposal, batch_size=400):
    """Compute mean and stddev of optimal SD under proposal."""
    def optimal_grad(x):
        div = 2*lambda_reg
        return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x) / div

    @partial(jit, static_argnums=1)
    def compute_sd(samples, fun):
        return stein.stein_discrepancy(samples, target.logpdf, fun)

    def get_sds(key, n_samples, fun):
        sds = []
        for subkey in random.split(key, 100):
            samples = proposal.sample(n_samples, key=subkey)
            sds.append(compute_sd(samples, fun))
        return sds

    sds_optimal = get_sds(key, batch_size, optimal_grad)
    return np.mean(sds_optimal), np.std(sds_optimal)


rain_drops = onp.zeros(5, dtype=[('position', float, 2),
                                      ('size',     float, 1),
                                      ('growth',   float, 1),
                                      ('color',    float, 4)])

rain_drops['position']



target = distributions.GaussianMixture()
circle_mix = distributions.Setup


from models import Particles


setup = distributions.Setup(target=distributions.Gaussian(0, 16), proposal=distributions.Gaussian(0, 1))
# setup = distributions.double_mixture
# setup = distributions.banana_target
target, proposal = setup.get()
kernel = kernels.get_rbf_kernel(1)

def score_est_unnormalized(x, inducing_particles):
    """kernel-smoothed estimate of grad(log q(x))"""
    return -np.mean(vmap(grad(kernel), (0, None))(inducing_particles, x))

def phistar_unnormalized(x, inducing_particles, aux=False):
    out = stein.phistar_i(x, inducing_particles, target.logpdf, kernel, aux=aux)
    if aux:
        neg_dKL, auxdata = out
        out = (-neg_dKL, auxdata)
    else:
        out = -out
    return out

## Normalize score estimate
s = proposal.sample(10_000)
inducing_particles = proposal.sample(10_000)
l2_score     = utils.l2_norm(s, grad(proposal.logpdf))
l2_score_est = utils.l2_norm(s, lambda x: score_est_unnormalized(x, inducing_particles))

def score_est(x, inducing_particles):
    """kernel-smoothed estimate of grad(log q(x)). Normalized."""
    return score_est_unnormalized(x, inducing_particles) * l2_score/l2_score_est

## Normalize kernel
def grad_kl(x):
    """True grad(KL)"""
    return setup.grad_kl(x)

l2_grad_kl = utils.l2_norm(np.squeeze(s), grad_kl)
l2_phistar = utils.l2_norm(s, lambda x: phistar_unnormalized(x, inducing_particles))

def phistar(x, inducing_particles, aux):
    if aux:
        out = phistar_unnormalized(x, inducing_particles, aux)
        return [o * l2_grad_kl/l2_phistar for o in out]
    else:
        return phistar_unnormalized(x, inducing_particles, aux) * l2_grad_kl/l2_phistar


def dKL(x, inducing_particles):
    """Estimate of grad(KL(x))"""
    dKL = score_est(x, inducing_particles) - grad(target.logpdf)(x)
    return dKL


def dKL_batched(particles, inducing_particles, aux=False):
    out = vmap(dKL, (0, None))(particles, inducing_particles)
    return (out, _) if aux else out


def phistar_batched(particles, inducing_particles, aux=False):
    return vmap(phistar, (0, None, None))(particles, inducing_particles, aux)


inducing_particles = proposal.sample(1000)
x = proposal.sample(1000)

xgrid = x.sort(axis=0)

fig, axs = plt.subplots(2,2, figsize=[16, 8])
axsiter = iter(axs.flatten())
xlims = (-8, 12)

ax = next(axsiter)
setup.plot(ax=ax, lims=xlims)
ax.legend()

ax = next(axsiter)
ax.plot(xgrid, dKL_batched(xgrid, inducing_particles), label="kernelized score matching grad(KL)")
plot.plot_fun(grad_kl, lims=xlims, ax=ax, label="True grad(KL)")
ax.legend()

ax = next(axsiter)
ax.plot(xgrid, phistar_batched(xgrid, inducing_particles), label="SVGD")
plot.plot_fun(grad_kl, lims=xlims, ax=ax, label="True grad(KL)")
ax.legend()

ax = next(axsiter)

for ax in axs.flatten():
    ax.set_xlim(xlims)


particles_svgd = Particles(key,
                           phistar_batched,
                           proposal,
                           learning_rate=0.01)

particles_score = Particles(key,
                            dKL_batched,
                            proposal,
                            learning_rate=0.01)


for _ in range(2500):
    inducing_particles = particles_svgd.get_params()
    particles_svgd.step(params=inducing_particles)

# _ = plt.hist(particles_svgd.get_params()[:,0], bins=25, density=True, alpha=0.5)
# _ = plot.plot_fun(target.pdf, lims=(-10, 10))


plt.plot(particles_svgd.rundata["std"], label="SVGD")


sdfkj


string_f1 = "kxkjdf![img](/home/lauro/obsidian/Pasted image 14.png)xdkfjdlk\n"
string_f2 = "![[Pasted image 14.png]]sdlfsdfsdfj\n"


def f1_to_f2(string):
    ind_begin = string.rindex("/") # last occurence of /
    ind_end = string.rindex(")") # last occurence of )
    name = string[ind_begin+1:ind_end]
    return f"![[{name}]]"

def f2_to_f1(string):
    ind_begin = string.index("!")
    ind_end = string.rindex("]")
    name = string[ind_begin+3:ind_end-1]
    return f"![img](/home/lauro/obsidian/{name})"

def line_is_f1_img(line):
    return line.startswith("![img]") and line.endswith(".png)\n")

def line_is_f2_img(line):
    return line.startswith("![[") and line.endswith(".png]]\n")


def convert_file(filename, direction="f1_to_f2"):
    if direction == "f1_to_f2":
        convert = f1_to_f2
        is_img = line_is_f1_img
    elif direction == "f2_to_f1":
        convert = f2_to_f1
        is_img = line_is_f2_img
    else:
        raise ValueError()
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if is_img(line):
                line_converted = convert(line)
                lines[i] = line_converted + "\n"
                print(f"changed line: {line}")
    with open(filename, "w") as f:
        f.writelines(lines)


# filename = "/home/lauro/testfile"
# convert_file(filename, direction="f1_to_f2")


# filename = "/home/lauro/obsidian/Master thesis/Updates/Update September 8.md"


convert_file(filename, direction="f1_to_f2")
