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


def neural_score_flow(key,
                      setup,
                      n_particles=300,
                      n_steps=1000,
                      sizes=[32, 32, 1],
                      particle_lr=1e-1,
                      learner_lr=1e-2,
                      lambda_reg=1/2,
                      noise_level=0):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()
    score_learner = models.ScoreLearner(key=keya,
                                       target=target,
                                       sizes=sizes,
                                       learning_rate=learner_lr,
                                       lambda_reg=lambda_reg,
                                       patience=25,
                                       lam=0.05)

    score_particles = models.Particles(key=keya,
                                         gradient=score_learner.gradient,
                                         proposal=proposal,
                                         n_particles=n_particles,
                                         learning_rate=particle_lr)

    for i in tqdm(range(n_steps), disable=False):
        noise = noise_level * 0.995**(1 + i)
        key, subkey = random.split(key)
        score_particles.reshuffle_tv()
        train_x, val_x, _ = score_particles.get_params(split_by_group=True)
        score_learner.train(train_x, val_x, key=subkey, n_steps=500, noise_level=noise)
        score_particles.step(score_learner.get_params(), noise_pre=noise)

    return score_learner, score_particles

def neural_svgd_flow(key,
                     setup,
                     n_particles=300,
                     n_steps=1000,
                     sizes=[32, 32, 1],
                     particle_lr=1e-1,
                     learner_lr=1e-2,
                     lambda_reg=1/2,
                     noise_level=0):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()
    learner = models.SDLearner(key=keya,
                               target=target,
                               sizes=sizes,
                               learning_rate=learner_lr,
                               lambda_reg=lambda_reg,
                               patience=15)

    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 proposal=proposal,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr)

    for _ in tqdm(range(n_steps)):
        try:
            key, subkey = random.split(key)
            particles.reshuffle_tv()
            train_x, val_x, _ = particles.get_params(split_by_group=True)
            learner.train(train_x, val_x, key=subkey, n_steps=500, noise_level=noise_level)
            particles.step(learner.get_params(), noise_pre=noise_level)
        except FloatingPointError as err:
            return learner, particles, err
    return learner, particles, None

def svgd_flow(key,
              setup,
              n_particles=300,
              n_steps=1000,
              particle_lr=1e-1,
              lambda_reg=1/2,
              noise_level=0,
              scaled=False):
    key, keya, keyb = random.split(key, 3)
    target, proposal = setup.get()

    kernel_gradient = models.KernelGradient(target=target,
                                            key=keya,
                                           lambda_reg=lambda_reg)
    gradient = partial(kernel_gradient.gradient, scaled=scaled) # scale to match lambda_reg

    svgd_particles = models.Particles(key=keyb,
                                     gradient=kernel_gradient.gradient,
                                     proposal=proposal,
                                     n_particles=n_particles,
                                     learning_rate=particle_lr)

    for _ in tqdm(range(n_steps)):
        svgd_particles.step(None, noise_pre=noise_level)

    return kernel_gradient, svgd_particles
