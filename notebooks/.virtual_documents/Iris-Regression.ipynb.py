get_ipython().run_line_magic("load_ext", " autoreload")
import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")

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

key = random.PRNGKey(0)


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


iris = datasets.load_iris()
features, labels = iris['data'], iris['target']

num_features = features.shape[-1]
num_classes = len(iris.target_names)


Root = tfd.JointDistributionCoroutine.Root
def model():
    w = yield Root(tfd.Sample(tfd.Normal(0., 1.),
                            sample_shape=(num_features, num_classes), name="w"))
    b = yield Root(tfd.Sample(tfd.Normal(0., 1.), sample_shape=(num_classes,), name="b"))
    logits = jnp.dot(features, w) + b
    _ = yield tfd.Independent(tfd.Categorical(logits=logits), reinterpreted_batch_ndims=1, name="labels") # this is why log_prob(sample) doesnt work: the batch dims are switched


dist = tfd.JointDistributionCoroutine(model)
def target_log_prob(*params):
    return dist.log_prob(params + (labels,))


# for intuition on how the model works:
key, subkey = random.split(key)
w = random.normal(subkey, shape=(num_features, num_classes))
key, subkey = random.split(key)
b = random.normal(subkey, shape=(num_classes, ))

dist.log_prob((w, b, labels))


init_key, sample_key = random.split(random.PRNGKey(0))
init_params = tuple(dist.sample(seed=init_key)[:-1])

@jit
def run_nuts_chain(key, state):
    kernel = tfp.mcmc.NoUTurnSampler(target_log_prob, 1e-3)
    return tfp.mcmc.sample_chain(500,
      current_state=state,
      kernel=kernel,
      trace_fn=lambda _, results: results.target_log_prob,
      num_burnin_steps=500, # CHANGED
      seed=key)

states, log_probs = run_nuts_chain(sample_key, init_params)
plt.figure()
plt.plot(log_probs)
plt.ylabel('Target Log Prob')
plt.xlabel('Iterations of NUTS')
plt.show()


params = dist.sample(seed=key)
dists, _ = dist.sample_distributions(seed=random.PRNGKey(0),
                                   value=params + (None,))


dists[-1].distribution.probs_parameter().shape


dists[-1].distribution.probs_parameter


def classifier_probs(params):
    dists, _ = dist.sample_distributions(seed=random.PRNGKey(0),
                                       value=params + (None,))
    return dists[-1].distribution.probs_parameter() # ie pdf of categorical(logit)


all_probs = jit(vmap(classifier_probs))(states)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


kernel = tfp.mcmc.UncalibratedLangevin(target_log_prob, 1e-3)


@jit
def run_lmc_chain(key, state):
    kernel = tfp.mcmc.UncalibratedLangevin(target_log_prob, 1e-3)
    return tfp.mcmc.sample_chain(500,
      current_state=state,
      kernel=kernel,
      trace_fn=lambda _, results: results.target_log_prob,
      num_burnin_steps=500, # CHANGED
      seed=key)

states, log_probs = run_lmc_chain(sample_key, init_params)
plt.figure()
plt.plot(log_probs)
plt.ylabel('Target Log Prob')
plt.xlabel('Iterations of LMC')
plt.show()


n_particles = 500
keys = random.split(key, n_particles)
init_batch = dist.sample(n_particles, seed=key)[:-1]

# NUTS
nuts_states, nuts_logp = vmap(run_nuts_chain)(keys, init_batch)
final_nuts_states = [param[:, -1, :] for param in nuts_states]

# LMC
lmc_states, lmc_logp = vmap(run_lmc_chain)(keys, init_batch)
final_lmc_states = [param[:, -1, :] for param in lmc_states]


plt.plot(np.rollaxis(nuts_logp, 1), color="green", label="NUTS")
plt.plot(np.rollaxis(lmc_logp, 1), color="red", label="LMC");
# plt.legend()


params = dist.sample(seed=key)[:-1]
params_flat, unravel = jax.flatten_util.ravel_pytree(params)
# unravel(params_flat) == params

batch = dist.sample(3, seed=key)[:-1]
batch_flat = batch_ravel(batch)
# batch == batch_unravel(batch_flat)


def logp(params_flat):
    return target_log_prob(*unravel(params_flat))

def ravel(params):
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    return flat

def batch_ravel(batch):
    return vmap(ravel)(batch)

def batch_unravel(batch_flat):
    return vmap(unravel)(batch_flat)


def run_svgd(key, init_batch):
    """init_batch is a batch of initial samples / particles."""
    init_batch = batch_ravel(init_batch)
    key, keya, keyb = random.split(key, 3)
    kernel_gradient = models.KernelGradient(target_logp=logp,
                                            key=keya)
    gradient = partial(kernel_gradient.gradient, scaled=True) # scale to match lambda_reg

    svgd_particles = models.Particles(key=keyb,
                                      gradient=gradient,
                                      init_samples=init_batch,
                                      learning_rate=1e-3,
                                      num_groups=1)
    for _ in range(500):
        svgd_particles.step(None)
    return batch_unravel(svgd_particles.particles.training), kernel_gradient, svgd_particles


def run_lmc(key, init_batch):
    """init_batch is a batch of initial samples / particles."""
    init_batch = batch_ravel(init_batch)
    key, keya, keyb = random.split(key, 3)
    energy_gradient = models.EnergyGradient(target_logp=logp, key=keya)

    lmc_particles = models.Particles(key=keyb,
                                     gradient=energy_gradient.gradient,
                                     init_samples=init_batch,
                                     learning_rate=1e-3,
                                     num_groups=1,
                                     noise_level=1.)
    for _ in range(500):
        lmc_particles.step(None)
    return batch_unravel(lmc_particles.particles.training), energy_gradient, lmc_particles


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
    for _ in range(1000):
        key, subkey = random.split(key)
        learner.train(next_batch, key=subkey, n_steps=1)
        particles.step(learner.get_params())
    return batch_unravel(particles.particles.training), learner, particles


init_batch = dist.sample(n_particles, seed=key)[:-1]
svgd_samples, gradient, particles = run_svgd(key, init_batch)
lmc_samples, lmc_gradient, lmc_particles = run_lmc(key, init_batch)
neural_samples, neural_gradient, neural_particles = run_neural_svgd(key, init_batch)


all_probs.shape


# Neural samples
all_probs = jit(vmap(classifier_probs))(neural_samples)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


plt.plot(neural_gradient.rundata["training_sd"])
plt.plot(neural_gradient.rundata["validation_sd"])


# NUTS samples
all_probs = jit(vmap(classifier_probs))(states)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# Parallel NUTS final state
all_probs = jit(vmap(classifier_probs))(tuple(final_nuts_states))
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# Parallel LMC (TFP) samples
all_probs = jit(vmap(classifier_probs))(tuple(final_lmc_states))
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# Parallel LMC (Lauro) samples
all_probs = jit(vmap(classifier_probs))(lmc_samples)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# SVGD samples
all_probs = jit(vmap(classifier_probs))(svgd_samples)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# random samples
all_probs = jit(vmap(classifier_probs))(tuple(dist.sample(100, seed=key)[:-1]))
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# remember, we have final states of vmapped tfp samplers:
w_nuts, b_nuts = final_nuts_states # all shaped (100, 3)
w_lmc, b_lmc = final_lmc_states

# as well as those from my methods
w_svgd, b_svgd = svgd_samples
w_lmc_l, b_lmc_l = lmc_samples
w_neural, b_neural = neural_samples


get_ipython().run_line_magic("matplotlib", " inline")
fig = plt.figure(figsize=[13, 13])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*np.rollaxis(b_nuts, 1), label="NUTS")
ax.scatter(*np.rollaxis(b_svgd, 1), label="SVGD")
ax.scatter(*np.rollaxis(b_lmc, 1), label="LMC")
ax.scatter(*np.rollaxis(b_lmc_l, 1), label="LMC Lauro")
ax.legend()


def e_score(samples):
    """Should return c, where true_logp = logp + c"""
    return np.mean(vmap(target_log_prob)(*samples))


sample_categories = [
    final_lmc_states,
    final_nuts_states,
    svgd_samples,
    lmc_samples,
]


for sample in sample_categories:
    print(e_score(sample))


svgd_samples_flat = batch_ravel(svgd_samples)


svgd_samples_flat.shape



