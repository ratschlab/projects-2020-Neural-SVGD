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


class BayesLR:
    def __init__(X, Y, a0, b0):
        self.X, self.Y = X, Y
        self.a0, self.b0 = a0, b0
    
    def dlnprob(self, theta):
        Xs = self.X[ridx, :] # batches
        Ys = self.Y[ridx]

        w = theta[:, :-1]  # logistic weights
        alpha = np.exp(theta[:, -1])  # the last column is logalpha
        d = w.shape[1]

        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1))

        coff = np.matmul(Xs, w.T)
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))

        dw_data = np.matmul(((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T, Xs)  # Y \in {-1,1}
        dw_prior = -np.multiply(nm.repmat(np.vstack(alpha), 1, d) , w)
        dw = dw_data * 1.0 * self.X.shape[0] / Xs.shape[0] + dw_prior  # re-scale

        dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha + 1  # the last term is the jacobian term

        return np.hstack([dw, np.vstack(dalpha)])  # % first order derivative 


a0, b0 = 1, 0.01 # hyper-parameters
t = tfd.Gamma(a0, b0, validate_args=True)
gamma_pdf = lambda x: jax.scipy.stats.gamma.pdf(x, a0, 0, 1/b0)
plot.plot_fun(t.prob, lims=(-10,500))
plot.plot_fun(gamma_pdf, lims=(-10, 500))


dist = tfd.ExpGamma(a0, b0, validate_args=True)
dist0 = tfd.Gamma(a0, b0, validate_args=True)


key, subkey = random.split(key)
s = dist.sample(1000, seed=subkey)
key, subkey = random.split(key)
s0 = dist0.sample(1000, seed=subkey)


plt.hist(np.exp(s), density=True, bins=50, alpha=.5)
plt.hist(s0, density=True, bins=50, alpha=.5)
plot.plot_fun(dist0.prob, lims=(0, 500))


logp = dist.log_prob
key, subkey = random.split(key)
init_particles = random.normal(subkey, shape=(100,))


def run_lmc(key, particles):
    """init_batch is a batch of initial samples / particles."""
    eta = 1e-1
    logps = []

    @jit
    def step(key, particles):
        log_probs, grads = vmap(value_and_grad(logp))(particles)
        particles += eta * grads + np.sqrt(2*eta) * random.normal(key, shape=particles.shape)
        return particles, log_probs

    for _ in tqdm(range(1000)):
        key, subkey = random.split(key)
        particles, log_probs = step(subkey, particles)
        logps.append(log_probs)
    return particles, np.array(logps)


key, subkey = random.split(key)
particles, logps = run_lmc(subkey, init_particles)


plt.plot(logps.mean(axis=1))


plt.hist(np.exp(particles), density=True)
plot.plot_fun(dist0.prob, lims=(0, 500))


np.all(np.isfinite(particles))


from dataclasses import dataclass


@dataclass
class Test:
    a: int
    b: int
t = Test(1,2)
t


t.


get_ipython().run_line_magic("autoreload", "")


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


target = distributions.Gaussian([0, 0], 1)


n = 300
init_key, sample_key = random.split(random.PRNGKey(0))
init_params = random.normal(key, shape=(n, 2))-5

@jit
def run_chain(key, state):
    kernel = tfp.mcmc.UncalibratedLangevin(target.logpdf, 1e-2)
    return tfp.mcmc.sample_chain(500,
      current_state=state,
      kernel=kernel,
      trace_fn=lambda _, results: results.target_log_prob,
      num_burnin_steps=0, # CHANGED
      seed=key)

subkeys = random.split(sample_key, n)
states, log_probs = vmap(run_chain)(subkeys, init_params)


states.shape


means = onp.mean(states, axis=(0, 2))
plt.plot(means)


_ = plt.hist(states[:, 500, :].flatten(), bins=25, density=True)
plt.hist(target.sample(500).flatten(), alpha=0.5, density=True);


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


# for comparison
banana = distributions.Banana([0, 0], [4, 1]) # ie y = x**2 + eps; std 2 and 1 respectively

tfp_banana = tfd.JointDistributionSequential([
    tfd.Normal(0, 2),
    lambda x: tfd.Normal(x**2, 1)
])


plt.scatter(*tfp_banana.sample(1000, seed=random.PRNGKey(0)), alpha=0.5)
plot.scatter(banana.sample(1000, key=random.PRNGKey(0)), alpha=0.5)


x = np.array([0, 0])
print(banana.logpdf(x))
print(tfp_banana.log_prob(x))


Root = tfd.JointDistributionCoroutine.Root
def model():
    x = yield Root(tfd.Normal(0, 2, name="x"))
    y = yield tfd.Normal(x**2, 1, name="y")
banana_coroutine = tfd.JointDistributionCoroutine(model)

banana_coroutine.sample(seed=random.PRNGKey(0))


xdist.distribution.sample(1, seed=key)


gen = model()
xroot = next(gen)
x = xroot.distribution.sample(1, key)
ydist = gen.send(x)
y = ydist.sample(1, key)
print(x, y)


samples = banana_coroutine.sample(1000, seed=key)
plt.scatter(samples.x, samples.y)


x = np.array([0, 0])
banana_coroutine.log_prob(x) # x = [x, y]


# d = distributions.GaussianMixture()
d = 2
coords = np.vstack([np.eye(d), np.ones(d).reshape(1, d)]) * 10

# sample m points from the corners of n-simplex

m = 3
idx = random.choice(key, d+1, (m,), replace=False)
means = coords[idx]

target = distributions.GaussianMixture(means, 1, np.ones(m))
proposal = distributions.Gaussian(-np.ones(d), np.ones(d))
setup = distributions.Setup(target, proposal)


n = 500
s = target.sample(n)
plt.scatter(*np.rollaxis(s, 1))


p = proposal.sample(100)


fig, ax = plt.subplots()
ax.set(xlim=lims, ylim=lims)
plot.scatter(p)

eta = 1e-1
key, subkey = random.split(key)
p = p + eta * vmap(grad(target.logpdf))(p) + np.sqrt(2*eta) * random.normal(subkey, p.shape)


# get coordinates of d-simplex
d = 5
coords = np.vstack([np.eye(d), np.ones(d).reshape(1, d)]) * 5

# sample m points
m = d+1
idx = random.choice(key, d+1, (m,), replace=False)
means = coords[idx]

target = distributions.GaussianMixture(means, 1, np.ones(m))
proposal = distributions.Gaussian(np.zeros(d), np.ones(d))


# rotated gaussian
d = 2
variances = np.logspace(-2, 1, num=d)
subkey = random.PRNGKey(0)
Q = utils.qmult(subkey, d)
cov = Q.T @ np.diag(variances) @ Q
rot_gauss = distributions.Gaussian(np.zeros(d), cov)
gauss = distributions.Gaussian(np.zeros(d), variances)


plot.plot_fun_2d(gauss.pdf)


plot.plot_fun_2d(rot_gauss.pdf)


skdfjlka



d = distributions.Gaussian(0, 1)
plot.plot_fun(grad(d.logpdf))


plt.subplots(figsize=[7,7])
d = distributions.Gaussian([0, 0], [1, 4])
plot.plot_gradient_field(vmap(grad(d.logpdf)))


def net_fn(x):
    net = hk.nets.MLP(output_sizes=[32, 32, 2],
                  w_init=hk.initializers.VarianceScaling(scale=2.0),
                  activation=jax.nn.swish,
                  activate_final=False)
    return net(x)
net = hk.transform(net_fn)


n = 100
d = 3
key, subkey = random.split(key)
x = random.normal(subkey, (n, d))

key, subkey = random.split(key)
params = net.init(subkey, x)
xout = net.apply(params, None, x)
plot.scatter(xout)
plot.scatter(distributions.Gaussian([0, 0], [1, 1]).sample(100))
lim=(-4, 4)
plt.ylim(lim)
plt.xlim(lim)


# define kernel
def kernel_fn(x, y):
    k = nets.DeepKernel([2, 2, 2])
#     k = nets.RBFKernel(parametrization="full")
    return k(np.stack([x, y]))
deep_kernel = hk.transform(kernel_fn)

# init params
d = 3
x = np.ones(d)
params = deep_kernel.init(subkey, x, x)
def k(x, y):
    return deep_kernel.apply(params, None, x, y)

# evaluate at particles
n = 50
samples = random.normal(subkey, (n, d))
kvalues = vmap(k, (None, 0))(samples[0], samples[1:])

plt.hist(kvalues)


np.count_nonzero(kvalues > 1e-1)


sigma = np.diag(np.ones(2))
k = kernels.get_multivariate_gaussian_kernel(sigma)
dist2 = distributions.Gaussian([0, 0], [1, 1])
x, y = dist2.sample(2)
k(x, y)





k = kernels.get_rbf_kernel(1)
kh = kernels.get_rbf_kernel(3)
n = 1000
samples_d2 = random.normal(subkey, shape=(n, 2))
samples_d20 = random.normal(subkey, shape=(n, 20))


kvalues_d2 =  vmap(k, (None, 0))(samples_d2[0], samples_d2[1:])
kvalues_d20 = vmap(kh, (None, 0))(samples_d20[0], samples_d20[1:])


_ = plt.hist(np.log10(kvalues_d2), bins=20, alpha=0.5, label="Dimension 2", density=True)
_ = plt.hist(np.log10(kvalues_d20), bins=20, alpha=0.5, label="Dimension 20", density=True)
plt.legend()
# plt.axvline(x=-1.5, color="red", linestyle="--")


key, subkey = random.split(key)
def mean_k()
permuted_samples = random.permutation(subkey, samples_d2)
mean_k_2d = np.mean(vmap(k)(samples_d2, permuted_samples))

key, subkey = random.split(key)
permuted_samples = random.permutation(subkey, samples_d20)
mean_k_20d = np.mean(vmap(k)(samples_d20, permuted_samples))


print(mean_k_2d)
print(mean_k_20d)


sizes=[4, 4, 3]


d = 2
key, subkey = random.split(key)
x_dummy = random.normal(subkey, shape=(2,d))

key, subkey = random.split(key)
k = hk.transform(lambda x: nets.RBFKernel(parametrization="full")(x))
params = k.init(subkey, x_dummy)

key, subkey = random.split(key)
k2 = hk.transform(lambda x: nets.RBFKernel(scale_param=True)(x))
params2 = k2.init(subkey, x_dummy)


key, subkey = random.split(key)
xy = random.normal(subkey, shape=(2,d))
xy


kernel = kernels.get_rbf_kernel(1)
kernel(*xy)


k.apply(params, None, xy)


params2


k2.apply(params2, None, xy)


d = hk.data_structures.to_mutable_dict(params2)
d["rbf_kernel"]["scale"] = np.array(2.)


k2.apply(d, None, xy) == 2* k2.apply(params2, None, xy)


d = hk.data_structures.to_mutable_dict(params2)
d["rbf_kernel"]["bandwidth"] = np.array([2., 2.])


k2.apply(d, None, xy)# == 2* k2.apply(params2, None, xy)


key, subkey = random.split(key)
sigma = random.normal(key, shape=(2,2))
ard = kernels.get_ard_kernel(sigma, dim=2)

ard_f = lambda x: ard(x, [0.,0.])

plot.plot_fun_2d(ard_f)


class FooMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self.foo = 'foo'

class Bar:
    def __init__(self, bar):
        self.bar = bar

class FooBar(FooMixin, Bar):
    def __init__(self, bar='bar'):
        super().__init__(bar)  # a single call is enough to invoke
                               # all parent constructors


fb = FooBar('testbar')


class A:
    def __init__(self):
#         super().__init__()
        print("init A")

class B:
    def __init__(self):
        super().__init__()
        print("init B")

class C(A, B):
    def __init__(self):
        super().__init__()

c = C()


def vec_fn(x):
    vec = nets.VectorField([4,4,1])
    return vec(x)
v = hk.transform(vec_fn)


x = np.ones((1,))
params = v.init(key, x)


x = random.normal(key, (10,1))
out = v.apply(params, None, x)
out


sdlfj


class Norm(hk.Module):
    """Much simpler than using BatchNorm"""
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        """x is a batch of particles of shape (n, d)"""
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / (std + 1e-5)

def test_fn(x):
    norm = Norm()
    return norm(x)

t = hk.transform(test_fn)


x = random.normal(key, (5000,)) * 2 + 3
out = t.apply(None, None, x)


plt.hist(out, density=True, alpha=0.5, bins=30, label="Normalized")
_ = plt.hist(x, density=True, alpha=0.5, bins=30, label="Original")
plt.legend()


class Test(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        a = hk.get_parameter()
        return x * a

def t_fn(x):
    norm = hk.BatchNorm(False, False, 0)
    return norm(x, is_training=True)

t = hk.transform_with_state(t_fn)


key, subkey = random.split(key)
x_dummy = random.normal(subkey, shape=(2,))
params, state = t.init(key, x_dummy)


t.apply(params, state, key, x_dummy)


x = random.normal(subkey, shape=(1000,2))
out, state = t.apply(params, state, key, x)
out - x


state


target = distributions.Gaussian(0, 1)
proposal = distributions.Gaussian(-3, 1)

key, subkey = random.split(key)
gradient = models.EnergyGradient(target, subkey)
key, subkey = random.split(key)
particles = models.Particles(subkey, gradient.gradient, proposal, n_particles=50, num_groups=1, learning_rate=1e-1, optimizer="adam", noise_level=1.)


scales = []
for _ in range(100):
    particles.step(None)
    scales.append(onp.squeeze(onp.abs(particles.noise_scales)).tolist())


fig, axs = plt.subplots(2, figsize=[8, 6])
particles.plot_trajectories(marker=".", ax=axs[0])
axs[1].plot(scales, "--.")
plt.yscale("log")


fig, axs = plt.subplots(2, figsize=[8, 6])
particles.plot_trajectories(marker=".", ax=axs[0])
axs[1].plot(scales, "--.")
plt.yscale("log")


from distributions import Gaussian as G


get_ipython().run_line_magic("autoreload", "")


target = G(0,1)


learner = models.SDLearner(target)
particles = models.Particles(key, learner.gradient, target)


learner.train(particles.next_batch, key=subkey, n_steps=5)
particles.step(learner.get_params())


learner.rundata["train_steps"]


learner.step_counter


learner.rundata["step_counter"]


get_ipython().run_line_magic("autoreload", "")


def check_dist(dist, key):
    s = dist.sample(10**4, key=key)
    errs = {
        "se_mean": np.sum((np.mean(s, axis=0) - dist.mean)**2),
        "se_cov": np.sum((np.cov(s, rowvar=False) - dist.cov)**2),
    }
    for k, err in errs.items():
        if err > 1e-2:
            print(f"{k} to big! {k} = {err} > 0.01")
    return


dist = distributions.Gaussian(0, 1)
check_dist(dist, key)


dist, _ = distributions.funnel.get()
check_dist(dist, key) # fine, just hard to approximate


dist, _ = distributions.banana_target.get()
check_dist(dist, key) # fine


dist, _ = distributions.ring_target.get()
check_dist(dist, key) # fine


dist, _ = distributions.squiggle_target.get()
check_dist(dist, key)



