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
                     noise_level=None,
                     patience=default_patience,
                     aux=True,
                     compute_metrics=None):
    key, keya, keyb, keyc = random.split(key, 4)
    target, proposal = setup.get()
    learner = models.SDLearner(key=keya,
                               target_logp=target.logpdf,
                               target_dim=target.d,
                               sizes=sizes,
                               learning_rate=learner_lr,
                               patience=patience,
                               aux=aux)

    if compute_metrics is None:
        compute_metrics = metrics.get_mmd_tracer(
                                     target.sample(500, keyc))
    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 num_groups=1 if target.d > 2 else 2,
                                 optimizer="sgd",
                                 compute_metrics=compute_metrics)

    # Warmup
    def next_batch(key):
        return proposal.sample(2*n_particles, key).split(2)
    learner.train(
        next_batch=next_batch, n_steps=NUM_WARMUP_STEPS, early_stopping=False)

    n_learner_steps = 50
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            key, subkey = random.split(key)
            batch = particles.next_batch(subkey, batch_size=2*n_particles//3) # TODO set to 99??
            learner.train(batch, n_steps=n_learner_steps)
            particles.step(learner.get_params())
        except Exception as err:
            warnings.warn(f"Caught Exception")
            return learner, particles, err
    particles.done()
    return learner, particles, None


def svgd_flow(key,
              setup,
              n_particles=default_num_particles,
              n_steps=default_num_steps,
              particle_lr=1e-1,
              lambda_reg=1/2,
              noise_level=None,
              particle_optimizer="sgd",
              scaled=True,
              bandwidth=1.,
              compute_metrics=None):
    key, keyb, keyc = random.split(key, 3)
    target, proposal = setup.get()

    kernel_gradient = models.KernelGradient(target_logp=target.logpdf,
                                            kernel=kernels.get_rbf_kernel,
                                            bandwidth=bandwidth,
                                            lambda_reg=lambda_reg,
                                            scaled=scaled)


    if compute_metrics is None:
        compute_metrics = metrics.get_mmd_tracer(
                                     target.sample(500, keyc))
    svgd_particles = models.Particles(key=keyb,
                                      gradient=kernel_gradient.gradient,
                                      init_samples=proposal.sample,
                                      n_particles=n_particles,
                                      learning_rate=particle_lr,
                                      optimizer=particle_optimizer,
                                      compute_metrics=compute_metrics)
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            svgd_particles.step(None)
        except Exception as err:
            warnings.warn("caught error!")
            return kernel_gradient, svgd_particles, err
    svgd_particles.done()
    return kernel_gradient, svgd_particles, None


def sgld_flow(key,
              setup,
              n_particles=default_num_particles,
              n_steps=default_num_steps,
              particle_lr=1e-2,
              lambda_reg=1/2,
              noise_level=None,
              particle_optimizer="sgld",
              compute_metrics=None):
    keya, keyb, keyc = random.split(key, 3)
    target, proposal = setup.get()
    energy_gradient = models.EnergyGradient(target.logpdf, keya, lambda_reg=lambda_reg)

    if compute_metrics is None:
        compute_metrics = metrics.get_mmd_tracer(
                                     target.sample(500, keyc))
    particles = models.Particles(key=keyb,
                                 gradient=energy_gradient.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 optimizer=particle_optimizer,
                                 num_groups=1,
                                 compute_metrics=compute_metrics)
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            particles.step(None)
        except Exception as err:
            warnings.warn("Caught and returned exception")
            return energy_gradient, particles, err
    particles.done()
    return energy_gradient, particles, None
