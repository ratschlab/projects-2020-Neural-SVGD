get_ipython().run_line_magic("load_ext", " autoreload")
import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
from tqdm import tqdm

import jax.numpy as jnp
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
from itertools import cycle, islice
    
key = random.PRNGKey(0)


from sklearn.model_selection import train_test_split


from functools import partial
import kernels


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
sns.set(style='white')


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


covtype = datasets.fetch_covtype()
features, labels = covtype['data'], covtype['target']

num_features = features.shape[-1]
num_classes = len(np.unique(labels))


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


def get_batches(x, y, n_steps=500, batch_size=300):
    """Split x and y into batches"""
    assert len(x) == len(y)
    batch_cycle = cycle(zip(*[np.array_split(data, len(data)//batch_size) for data in (x, y)]))
    return islice(batch_cycle, n_steps)


num_batches = len(x_train) // 300
print("num batches:", len(x_train) // 300)


for a, b in get_batches(x_train, y_train, 5):
    print(a.shape)
    print(b.shape, "\n")


batches = get_batches(x_train, y_train, batch_size=5)


Root = tfd.JointDistributionCoroutine.Root

def get_model(features_batch):
    def model():
        """generator"""
        w = yield Root(tfd.Sample(tfd.Normal(0., 1.), sample_shape=(num_features, num_classes), name="w"))
        b = yield Root(tfd.Sample(tfd.Normal(0., 1.), sample_shape=(num_classes,),              name="b"))
        logits = jnp.dot(features_batch, w) + b
        _ = yield tfd.Independent(tfd.Categorical(logits=logits), reinterpreted_batch_ndims=1, name="labels")
    return model


def get_logp(x_batch, y_batch):
    """Stochastic estimate of the log-density (up to additive constant)
    based on batch"""
    def logp(params):
        dist = tfd.JointDistributionCoroutine(get_model(x_batch))
        return dist.log_prob(tuple(params) + (y_batch,))
    return logp

dist = tfd.JointDistributionCoroutine(get_model(x_train[:300]))


x, y = next(batches)
logp = get_logp(x, y)

dist = tfd.JointDistributionCoroutine(get_model(x))
*params, label = dist.sample(seed=key)
print(dist.log_prob(params + [y]))
print(logp(params))

# now batched!
*params, label = dist.sample(5, seed=key)
vmap(logp)(params)


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


def run_lmc(key, init_batch):
    """init_batch is a batch of initial samples / particles."""
    particles = batch_ravel(init_batch)
    eta = 1e-3
    logps = []

    @jit
    def step(key, particles):
        logp = get_flat_logp(x, y)
        log_probs, grads = vmap(value_and_grad(logp))(particles)
        particles += eta * grads + np.sqrt(2*eta) * random.normal(key, shape=particles.shape)
        return particles, log_probs

    for x, y in tqdm(get_batches(x_train, y_train, num_batches*2), total=num_batches*2):
        key, subkey = random.split(key)
        particles, log_probs = step(subkey, particles)
        logps.append(log_probs)
    return batch_unravel(particles), np.array(logps)


def run_svgd(key, init_batch):
    """init_batch is a batch of initial samples / particles."""
    init_batch = batch_ravel(init_batch)
    key, keya, keyb = random.split(key, 3)
    kernel_gradient = models.KernelGradient(target_logp=logp, key=keya)
    gradient = partial(kernel_gradient.gradient, scaled=True) # scale to match lambda_reg

    svgd_particles = models.Particles(key=keyb,
                                      gradient=gradient,
                                      init_samples=init_batch,
                                      learning_rate=1e-3,
                                      num_groups=1)
    for params, labels in get_batches(x_train, y_train, num_batches*2):
        svgd_particles.step(None)
    return batch_unravel(svgd_particles.particles.training), kernel_gradient, svgd_particles


def run_neural_svgd(key, init_batch):
    """init_batch is a batch of initial samples / particles."""
    init_batch = batch_ravel(init_batch)
    key, keya, keyb = random.split(key, 3)
    learner = models.SDLearner(target_logp=logp, target_dim=init_batch.shape[1], key=keya)

    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 init_samples=init_batch,
                                 learning_rate=1e-3,
                                 num_groups=2)
    next_batch = partial(particles.next_batch, batch_size=None)
    for x, y in get_batches(x_train, y_train, 1000):
        key, subkey = random.split(key)
        learner.train(next_batch, key=subkey, n_steps=1)
        particles.step(learner.get_params())
    return batch_unravel(particles.particles.training), learner, particles


init_batch = dist.sample(500, seed=key)[:-1]
lmc_samples, logps = run_lmc(key, init_batch)
# svgd_samples, gradient, particles = run_svgd(key, init_batch)
# neural_samples, neural_gradient, neural_particles = run_neural_svgd(key, init_batch)


logps.shape


plt.plot(logps.mean(axis=1));


params = tuple([p[0] for p in lmc_samples])


# get logits
test_dist = tfd.JointDistributionCoroutine(get_model(x_test))
@jit
def get_logits(params):
    """Returns logits shaped (n, 7), 7 being nr of categories"""
    dists, _ = test_dist.sample_distributions(seed=random.PRNGKey(0), value=params + (None,))
    logits = dists[-1].distribution.probs_parameter()
    return logits


# Parallel LMC (Lauro) samples
all_probs = vmap(get_logits)(lmc_samples)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == y_test))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == y_test))


# Random samples
all_probs = vmap(get_logits)(tuple(dist.sample(1000, seed=key)[:-1]))
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == y_test))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == y_test))


100/7
