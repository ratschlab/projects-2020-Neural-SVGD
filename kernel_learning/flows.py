import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, tree_util, jacfwd, grad
from jax.experimental import optimizers
from jax.ops import index_update, index
import haiku as hk
import jax
import numpy as onp
import matplotlib.pyplot as plt

import traceback
import time
import warnings
from tqdm import tqdm
from functools import partial
import json_tricks as json

import utils
import metrics
import stein
import kernels
import nets
import distributions
import plot
import models

default_num_particles = 50
default_num_steps = 100
#default_particle_lr = 1e-1
#default_learner_lr = 1e-2
default_noise_level = 0.
default_patience = 10
disable_tqdm = False
NUM_WARMUP_STEPS = 100

def neural_svgd_flow(key,
                     setup,
                     n_particles=default_num_particles,
                     n_steps=default_num_steps,
                     sizes=None,
                     particle_lr=1e-2,
                     learner_lr=1e-2,
                     noise_level=default_noise_level,
                     patience=default_patience,
                     aux=True):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()
    learner = models.SDLearner(key=keya,
                               target_logp=target.logpdf,
                               target_dim=target.d,
                               sizes=sizes,
                               learning_rate=learner_lr,
                               patience=patience,
                               aux=aux)

    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 noise_level=noise_level,
                                 num_groups=1 if target.d > 2 else 2,
                                 optimizer="sgd")

    # Warmup
    def next_batch(key):
        return proposal.sample(2*n_particles, key).split(2)
    learner.train(
        next_batch=next_batch, n_steps=NUM_WARMUP_STEPS, early_stopping=False)

    n_learner_steps = 50
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            key, subkey = random.split(key)
            batch = particles.next_batch(subkey, batch_size=2*n_particles//3)
            learner.train(batch, n_steps=n_learner_steps)
            particles.step(learner.get_params())
        except Exception as err:
            warnings.warn(f"Caught Exception")
            return learner, particles, err
    return learner, particles, None


def svgd_flow(key,
              setup,
              n_particles=default_num_particles,
              n_steps=default_num_steps,
              particle_lr=1e-1,
              lambda_reg=1/2,
              noise_level=default_noise_level,
              particle_optimizer="sgd",
              scaled=True,
              bandwidth=1.):
    key, keyb = random.split(key)
    target, proposal = setup.get()

    kernel_gradient = models.KernelGradient(target_logp=target.logpdf,
                                            kernel=kernels.get_rbf_kernel,
                                            bandwidth=bandwidth,
                                            lambda_reg=lambda_reg,
                                            scaled=scaled)
    svgd_particles = models.Particles(key=keyb,
                                      gradient=kernel_gradient.gradient,
                                      init_samples=proposal.sample,
                                      n_particles=n_particles,
                                      learning_rate=particle_lr,
                                      noise_level=noise_level,
                                      optimizer=particle_optimizer)
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            svgd_particles.step(None)
        except Exception as err:
            warnings.warn("caught error!")
            return kernel_gradient, svgd_particles, err
    return kernel_gradient, svgd_particles, None


def sgld_flow(key,
              setup,
              n_particles=default_num_particles,
              n_steps=default_num_steps,
              particle_lr=1e-2,
              lambda_reg=1/2,
              noise_level=1,
              particle_optimizer="sgd"):
    keya, keyb = random.split(key)
    target, proposal = setup.get()
    energy_gradient = models.EnergyGradient(target.logpdf, keya, lambda_reg=lambda_reg)
    particles = models.Particles(key=keyb,
                                 gradient=energy_gradient.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 optimizer=particle_optimizer,
                                 num_groups=1,
                                 noise_level=noise_level)
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            particles.step(None)
        except Exception as err:
            warnings.warn("Caught and returned exception")
            return energy_gradient, particles, err
    return energy_gradient, particles, None


def deep_kernel_flow(key,
                     setup,
                     n_particles=default_num_particles,
                     n_steps=default_num_steps,
                     sizes=[32, 32, 2],
                     particle_lr=1e-2,
                     learner_lr=1e-3,
                     noise_level=default_noise_level,
                     patience=default_patience):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()
    learner = models.KernelLearner(target_logp=target.logpdf,
                                   key=keya,
                                   sizes=sizes,
                                   learning_rate=learner_lr,
                                   patience=patience)

    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 noise_level=noise_level)

    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            key, subkey = random.split(key)
            learner.train(particles.next_batch, key=subkey, n_steps=1)
            particles.step(learner.get_params())
        except (FloatingPointError, KeyboardInterrupt) as err:
            warnings.warn(f"Caught floating point error")
            return learner, particles, err
    return learner, particles, None
