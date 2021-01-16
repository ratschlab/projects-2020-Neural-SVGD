import os
import json_tricks as json
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit, random
import numpy as onp

import distributions
import flows
import kernels
import metrics
import config as cfg

key = random.PRNGKey(0)
on_cluster = not os.getenv("HOME") == "/home/lauro"

# Config
NUM_STEPS = 500  # 500
PARTICLE_STEP_SIZE = 1e-2  # for particle update
LEARNING_RATE = 1e-4  # for neural network
NUM_PARTICLES = 200  # 200
MAX_DIM = 50  # sweep from 2 to MAX_DIM


mmd_kernel = kernels.get_rbf_kernel(1.)
mmd = jit(metrics.get_mmd(mmd_kernel))


def get_mmds(particle_list, ys):
    mmds = []
    for xs in [p.particles.training for p in particle_list]:
        mmds.append(mmd(xs, ys))
    return mmds


def sample(d, key, n_particles):
    target = distributions.Funnel(d)
    proposal = distributions.Gaussian(jnp.zeros(d), jnp.ones(d))
    funnel_setup = distributions.Setup(target, proposal)

    key, subkey = random.split(key)
    neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, funnel_setup, n_particles=n_particles, n_steps=NUM_STEPS, particle_lr=PARTICLE_STEP_SIZE, noise_level=0., patience=0, learner_lr=LEARNING_RATE)
    svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=NUM_STEPS, particle_lr=PARTICLE_STEP_SIZE, noise_level=0., scaled=True,  bandwidth=None)
    sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=NUM_STEPS, particle_lr=PARTICLE_STEP_SIZE, noise_level=1.)
    return (neural_particles, svgd_particles, sgld_particles), (neural_learner, svgd_gradient, sgld_gradient)


mmd_sweep = []
for d in tqdm(range(2, 40), disable=on_cluster):
    print(d)
    key, subkey = random.split(key)
    particles, gradients = sample(d, subkey, NUM_PARTICLES)

    target = distributions.Funnel(d)
    key, subkey = random.split(key)
    ys = target.sample(NUM_PARTICLES, subkey)
    mmds = get_mmds(particles, ys)
    mmd_sweep.append(mmds)

    print("MMDs:", mmds)
    print()
mmd_sweep = onp.array(mmd_sweep)

# save json results
results = {
    "NSVGD": mmd_sweep[:, 0].tolist(),
    "SVGD":  mmd_sweep[:, 1].tolist(),
    "SGLD":  mmd_sweep[:, 2].tolist(),
}

with open(cfg.results_path + "funnel-dimension-sweep.json", "w") as f:
    json.dump(results, f, indent=4, sort_keys=True, allow_nan=True)
