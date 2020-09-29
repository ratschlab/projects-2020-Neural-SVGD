get_ipython().run_line_magic("load_ext", " autoreload")
from jax import config
config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
import json_tricks as json
import copy
from functools import partial

from tqdm import tqdm
import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import matplotlib.pyplot as plt
import numpy as onp
import jax
import pandas as pd
import haiku as hk
import optax
from jax.experimental import optimizers

import config

import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import models
import flows

from jax.experimental import optimizers

key, subkey = random.split(random.PRNGKey(0))
from distributions import funnel, banana_target, ring_target, squiggle_target, mix_of_gauss


get_ipython().run_line_magic("autoreload", "")


noise_level = 1.
n_steps = 50
n_particles = 50 * 1 # careful: are you splitting into train / val / test sets??
setups = (funnel, banana_target, ring_target, squiggle_target, mix_of_gauss)
sgld_hparams = {
    "particle_lr": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "particle_optimizer": ["sgd", "adam"],
}
hparam_product = list(utils.dict_cartesian_product(**sgld_hparams))

# for d in hparam_product:
#     print(d)


# Structure:
# 1) call flow(target, hparams, etc)
# 2) compute final metrics on particles
# 3) save rundata to file


def run(hparams, setup):
    "Note: we're always using same random seed."
    t = time.time()
    startdate = time.strftime("%Y-%m-%d", time.localtime(t))
    starttime = time.strftime("%H:%M:%S", time.localtime(t))
    filename = f"/home/lauro/code/msc-thesis/svgd/runs/sgld/{startdate}_{starttime}.json"
    target, _ = setup.get()
    
    # 1) call flow
    gradient, particles, err =        flows.sgld_flow( subkey, setup, n_particles=n_particles, n_steps=n_steps, noise_level=noise_level, **hparams)
#     gradient, particles, err =        flows.svgd_flow( subkey, setup, n_particles=n_particles, n_steps=n_steps, noise_level=noise_level, scaled=True)
#     gradient, particles, err =        flows.score_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, noise_level=noise_level, scale=1.)

#     gradient, particles, err = flows.neural_svgd_flow( subkey, setup, n_particles=n_particles, n_steps=n_steps, noise_level=noise_level, learner_lr=1e-3, patience=10)
#     gradient, particles, err = flows.neural_score_flow(subkey, setup, n_particles=n_particles, n_steps=n_steps, noise_level=noise_level, learner_lr=1e-3, patience=10, lam=0)
#     gradient, particles, err = flows.deep_kernel_flow( subkey, setup, n_particles=n_particles, n_steps=n_steps, noise_level=noise_level, learner_lr=1e-3, patience=3)

    # 2) compute final metrics on particles
    final_particles = particles.rundata['particles'][-1]
    metrics_dict = metrics.compute_final_metrics(final_particles, target)

    # 3) save rundata
    data = {
        "hparams": hparams,
        "particle_data": particles.rundata,
        "gradient_data": gradient.rundata,
        "final_metrics": metrics_dict,
    }
    data = utils.dict_dejaxify(data)

    if os.path.exists(filename):
        filename = filename[:-5] + datetime.datetime.now().strftime(".%f.json")
    with open(filename, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)
    return gradient, particles, err


for setup in tqdm(setups):
    for hparams in hparam_product:
        gradient, particles, err = run(hparams, setup)
        if err is not None:
            raise err from None


directory = "/home/lauro/code/msc-thesis/svgd/runs/sgld/"


for filename in os.listdir(directory):
    print(filename)



