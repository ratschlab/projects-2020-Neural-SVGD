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
      num_burnin_steps=0, # CHANGED
      seed=key)

states, log_probs = run_nuts_chain(sample_key, init_params)
plt.figure()
plt.plot(log_probs)
plt.ylabel('Target Log Prob')
plt.xlabel('Iterations of NUTS')
plt.show()


def classifier_probs(params):
    dists, _ = dist.sample_distributions(seed=random.PRNGKey(0),
                                       value=params + (None,))
    return dists[-1].distribution.probs_parameter()


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
      num_burnin_steps=0, # CHANGED
      seed=key)

states, log_probs = run_lmc_chain(sample_key, init_params)
plt.figure()
plt.plot(log_probs)
plt.ylabel('Target Log Prob')
plt.xlabel('Iterations of LMC')
plt.show()


n = 3
keys = random.split(key, n)
init_batch = dist.sample(n, seed=key)[:-1]

# NUTS
nuts_states, nuts_logp = vmap(run_nuts_chain)(keys, init_batch)

# LMC
lmc_states, lmc_logp = vmap(run_lmc_chain)(keys, init_batch)


plt.plot(np.rollaxis(nuts_logp, 1), color="green", label="NUTS")
plt.plot(np.rollaxis(lmc_logp, 1), color="red", label="LMC")
plt.legend()


def logp(params_flat):
    return target_log_prob(*unravel(params_flat))

def ravel(params):
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    return flat

def batch_ravel(batch):
    return vmap(ravel)(batch)

def batch_unravel(batch_flat):
    return vmap(unravel)(batch_flat)


params = dist.sample(seed=key)[:-1]
params_flat, unravel = jax.flatten_util.ravel_pytree(params)
# unravel(params_flat) == params

batch = dist.sample(3, seed=key)[:-1]
batch_flat = batch_ravel(batch)
# batch == batch_unravel(batch_flat)


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
    energy_gradient = models.KernelGradient(target_logp=logp, key=keya)

    svgd_particles = models.Particles(key=keyb,
                                      gradient=gradient,
                                      init_samples=init_batch,
                                      learning_rate=1e-2,
                                      num_groups=1)
    for _ in range(500):
        svgd_particles.step(None)
        
    return batch_unravel(svgd_particles.particles.training), kernel_gradient, svgd_particles


init_batch = dist.sample(100, seed=key)[:-1]
params, gradient, particles = run_svgd(key, init_batch)


particles.n_particles


plt.plot(particles.rundata["training_logp"]);


plt.plot(np.rollaxis(nuts_logp, 1), color="green", label="NUTS")
plt.plot(np.rollaxis(lmc_logp, 1), color="red", label="LMC")


# NUTS samples
all_probs = jit(vmap(classifier_probs))(states)
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# SVGD samples
all_probs = jit(vmap(classifier_probs))(tuple(params))
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))


# random samples
all_probs = jit(vmap(classifier_probs))(tuple(dist.sample(100, seed=key)[:-1]))
print('Average accuracy:', jnp.mean(all_probs.argmax(axis=-1) == labels))
print('BMA accuracy:', jnp.mean(all_probs.mean(axis=0).argmax(axis=-1) == labels))



