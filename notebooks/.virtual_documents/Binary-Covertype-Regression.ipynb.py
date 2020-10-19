get_ipython().run_line_magic("load_ext", " autoreload")
import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
from tqdm import tqdm
from jax import config


import jax.numpy as jnp
import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax.ops import index_update, index
import matplotlib.pyplot as plt
import matplotlib
import numpy as onp
import jax
import pandas as pd
import scipy
import haiku as hk
    
import utils
import plot
import distributions
import stein
import models
import flows
from itertools import cycle, islice
    
key = random.PRNGKey(0)

from sklearn.model_selection import train_test_split

from functools import partial
import kernels

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
sns.set(style='white')

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


data = scipy.io.loadmat('/home/lauro/code/msc-thesis/wang_svgd/data/covertype.mat')
features = data['covtype'][:, 1:]
features = onp.hstack([features, onp.ones([features.shape[0], 1])]) # add intercept term

labels = data['covtype'][:, 0]
labels[labels == 2] = 0

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

num_features = features.shape[-1]
num_classes = len(onp.unique(labels))


def get_batches(x, y, n_steps=500, batch_size=100):
    """Split x and y into batches"""
    assert len(x) == len(y)
    n = len(x)
    idxs = onp.random.choice(n, size=(n_steps, batch_size))
    for idx in idxs:
        yield x[idx], y[idx]
#     batch_cycle = cycle(zip(*[onp.array_split(data, len(data)//batch_size) for data in (x, y)]))
#     return islice(batch_cycle, n_steps)

num_batches = len(x_train) // 100


# def get_batches(x, y, n_steps=500, batch_size=100):
#     """Split x and y into batches"""
#     assert len(x) == len(y)
#     batch_cycle = cycle(zip(*[onp.array_split(data, len(data)//batch_size) for data in (x, y)]))
#     return islice(batch_cycle, n_steps)

# num_batches = len(x_train) // 100


# batches = get_batches(x_train, y_train, batch_size=5)


a0, b0 = 1, 0.01 # hyper-parameters
t = tfd.Gamma(a0, b0, validate_args=True)
gamma_pdf = lambda x: jax.scipy.stats.gamma.pdf(x, a0, 0, 1/b0)
plot.plot_fun(t.prob, lims=(-10,100))
plot.plot_fun(gamma_pdf, lims=(-10, 100))


a0, b0 = 1, 0.01 # hyper-parameters
# note that b0 is inverse scale, so this means alpha big, so 1/alpha small! gaussian narrow! dunno why, check paper

Root = tfd.JointDistributionCoroutine.Root

def get_model(features_batch):
    def model():
        """generator"""
        log_alpha = yield Root(tfd.ExpGamma(a0, b0, name="log_alpha"))                                      # scalar
        w = yield tfd.Sample(tfd.Normal(0., 1/np.exp(log_alpha/2)), sample_shape=(num_features,), name="w") # shape (num_features,)
        log_odds = jnp.dot(features_batch, w)                                                               # shape (len(features_batch),)
        _ = yield tfd.Independent(tfd.Bernoulli(logits=log_odds), name="labels")                            # shape (len(features_batch),) in {0, 1}
    return model


def get_logp(x_batch, y_batch):
    """Stochastic estimate of the log-density (up to additive constant)
    based on batch"""
    def logp(params):
        dist = tfd.JointDistributionCoroutineAutoBatched(get_model(x_batch), validate_args=True)
        return dist.log_prob(tuple(params) + (y_batch,))
    return logp


dist = tfd.JointDistributionCoroutineAutoBatched(get_model(x_train[:7]))
key, subkey = random.split(key)
dist.sample(seed=subkey)


key, subkey = random.split(key)
_, w, labels = dist.sample(seed=subkey)
def pdf(log_alpha):
    return dist.prob((log_alpha, w, labels))


# plot.plot_fun(tfd.ExpGamma(a0, b0).prob, lims=(-10, 10))

plot.plot_fun(pdf, lims=(-10, 10))
log_alpha = np.log(np.var(w))
plt.axvline(x=-log_alpha) # makes sense, since 1/alpha \approx Var(w)


batches = get_batches(x_train, y_train, batch_size=5)
x, y = next(batches)
del batches
logp = get_logp(x, y)

dist = tfd.JointDistributionCoroutineAutoBatched(get_model(x), validate_args=True)
*params, labels = dist.sample(seed=key)
print(dist.log_prob(params + [y]))
print(logp(params))

# now batched!
key, subkey = random.split(key)
*params, labels = dist.sample(7, seed=subkey)
print(vmap(logp)(params))
print(logp(params))


params = dist.sample(seed=key)[:-1]
params_flat, unravel = jax.flatten_util.ravel_pytree(params)
# unravel(params_flat) == params
# [a == b for a, b in zip(unravel(params_flat), params)]


def get_flat_logp(x_batch, y_batch):
    logp = get_logp(x_batch, y_batch)
    def flat_logp(params_flat):
        return logp(unravel(params_flat))
    return flat_logp

def ravel(params):
    flat, _ = jax.flatten_util.ravel_pytree(params)
    return flat

def batch_ravel(batch):
    return vmap(ravel)(batch)

def batch_unravel(batch_flat):
    return vmap(unravel)(batch_flat)


def get_schedule(eta):
    def polynomial_schedule(step):
        return eta / (step + 1)**0.55
    def constant_schedule(step):
        return eta/5
    return constant_schedule

def get_probs(params):
    """
    Argument: sampled model parameters (single sample! need to vmap over sample batch)
    Returns logits shaped (n,)"""
    test_dist = tfd.JointDistributionCoroutineAutoBatched(get_model(x_test))
    dists, _ = test_dist.sample_distributions(seed=random.PRNGKey(0), value=params + (None,))
    probs = dists[-1].distribution.probs_parameter() # spits out probability of labels
    return probs

def get_preds(params):
    """
    Argument: sampled model parameters (batch)
    Returns predictions on test set
    """
    probs = vmap(get_probs)(params) # shape (n_samples, n_data)
    return (probs.mean(axis=0) > 0.5).astype(np.int32)

@jit
def test_accuracy(params):
    """
    Argument: sampled model parameters (batch)
    Returns accuracy on test set
    """
    return np.mean(get_preds(params) == y_test)


def run_lmc(key, init_batch):
    """init_batch is a batch of initial samples / particles."""
    particles = batch_ravel(init_batch)
    schedule = get_schedule(5e-3)
    logps = []
    accs = []

    @jit
    def step(key, particles, x, y, step_counter):
        eta = schedule(step_counter)
        logp = get_flat_logp(x, y)
        log_probs, grads = vmap(value_and_grad(logp))(particles)
        particles += eta * grads + np.sqrt(2*eta) * random.normal(key, shape=particles.shape)
        return particles, log_probs

    n_steps = num_batches*3
    for i, (x, y) in tqdm(enumerate(get_batches(x_train, y_train, n_steps+1)), total=n_steps):
        key, subkey = random.split(key)
        particles, log_probs = step(subkey, particles, x, y, i)
        logps.append(log_probs)
        if i % (n_steps//4)==0:
            accs.append(test_accuracy(batch_unravel(particles)))
    return batch_unravel(particles), {"logp": np.array(logps), "accuracy": np.array(accs)}


def run_svgd(init_batch):
    """init_batch is a batch of initial samples / particles."""
    particles = batch_ravel(init_batch)
    schedule = get_schedule(1e-1)
    logps = []
    accs = []

    @jit
    def step(particles, x, y, step_counter):
        eta = schedule(step_counter)
        logp = get_flat_logp(x, y)
        kernel = kernels.get_rbf_kernel(kernels.median_heuristic(particles))
        phistar = stein.get_phistar(kernel, logp, particles)
        log_probs = vmap(logp)(particles)
        return particles + eta * vmap(phistar)(particles), log_probs

    n_steps = num_batches*3
    for i, (x, y) in tqdm(enumerate(get_batches(x_train, y_train, n_steps+1)), total=n_steps):
        particles, log_probs = step(particles, x, y, i)
        logps.append(log_probs)
        if i % (n_steps//4)==0:
            accs.append(test_accuracy(batch_unravel(particles)))
    return batch_unravel(particles), {"logp": np.array(logps), "accuracy": np.array(accs)}


# def run_neural_svgd(key, init_batch):
#     """init_batch is a batch of initial samples / particles."""
#     particles = batch_ravel(init_batch)
#     schedule = get_schedule(5e-2)
#     logps = []
#     init_batch = batch_ravel(init_batch)
#     key, keya, keyb = random.split(key, 3)
#     learner = models.SDLearner(target_logp=logp, target_dim=init_batch.shape[1], key=keya)

#     particles = models.Particles(key=keyb,
#                                  gradient=learner.gradient,
#                                  init_samples=init_batch,
#                                  learning_rate=1e-3,
#                                  num_groups=2)
#     next_batch = partial(particles.next_batch, batch_size=None)
#     for x, y in get_batches(x_train, y_train, 10):
#         key, subkey = random.split(key)
#         learner.train(next_batch, key=subkey, n_steps=1)
#         particles.step(learner.get_params())
#     return batch_unravel(particles.particles.training), learner, particles


# Run samplers
init_batch = dist.sample(100, seed=key)[:-1]
lmc_samples, lmc_aux = run_lmc(key, init_batch)
svgd_samples, svgd_aux = run_svgd(init_batch)
# neural_samples, neural_gradient, neural_particles = run_neural_svgd(key, init_batch)


plt.plot(lmc_aux["accuracy"], "--.", label="SGLD")
plt.plot(svgd_aux["accuracy"], "--.", label="SVGD")
plt.legend()


plt.plot(lmc_aux["accuracy"], "--.", label="SGLD")
plt.plot(svgd_aux["accuracy"], "--.", label="SVGD")
plt.legend()


print("LMC accuracy:", test_accuracy(lmc_samples))
print("SVGD accuracy:", test_accuracy(svgd_samples))
key, subkey = random.split(key)
print("prior samples accuracy:", test_accuracy(dist.sample(500, seed=subkey)[:-1]))


test_batches = get_batches(x_test, y_test, batch_size=5000)


x, y = next(test_batches)
test_logp = get_logp(x, y)


plt.plot(lmc_aux["logp"].mean(axis=1), label="LMC")
plt.plot(svgd_aux["logp"].mean(axis=1), label="SVGD")
plt.legend()


lmc_test_logp = np.mean(vmap(test_logp)(lmc_samples))
svgd_test_logp = np.mean(vmap(test_logp)(svgd_samples))

fig, ax = plt.subplots(figsize=[8, 5])
# ax.plot(svgd_logps.mean(axis=1), label="SVGD")
ax.axhline(y=svgd_test_logp, label="SVGD test", color="tab:blue")

# ax.plot(lmc_logps.mean(axis=1), label="LMC")
ax.axhline(y=lmc_test_logp, label="LMC test", color="tab:orange")

plt.legend()






