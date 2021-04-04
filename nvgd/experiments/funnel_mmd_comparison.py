import json_tricks as json
import jax.numpy as jnp
from jax import random
from nvgd.src import distributions, flows, metrics
import config as cfg

key = random.PRNGKey(0)

# config
NUM_STEPS = 5000  # 5000
PARTICLE_STEP_SIZE = 1e-2
LEARNING_RATE = 1e-4
NUM_PARTICLES = 200  # 200
d = 2
PATIENCE = 15  # (-1 vs. 0 makes a noticeable diff, but not large)

target = distributions.Funnel(d)
proposal = distributions.Gaussian(jnp.zeros(d), jnp.ones(d))
funnel_setup = distributions.Setup(target, proposal)
target_samples = target.sample(NUM_PARTICLES)


key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(
    key=subkey,
    setup=funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE,
    patience=PATIENCE,
    learning_rate=LEARNING_RATE,
    aux=False,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False,
)

sgld_gradient, sgld_particles, err2 = flows.sgld_flow(
    subkey,
    funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False,
)

sgld_gradient2, sgld_particles2, err3 = flows.sgld_flow(
    subkey,
    funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE/5,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False
)

svgd_gradient, svgd_particles, err4 = flows.svgd_flow(
    subkey,
    funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE*5,
    scaled=True,
    bandwidth=None,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False,
)

# Note: I scaled the svgd step-size (by hand) so that it is maximial while
# still converging to a low MMD.

particle_containers = (neural_particles, sgld_particles,
                       sgld_particles2, svgd_particles)
names = ("Neural", "SGLD", "SGLD2", "SVGD")
results = {name: p.rundata["rbf_mmd"].tolist()  # CHANGED from 'funnel_mmd'
           for name, p in zip(names, particle_containers)}

with open(cfg.results_path + "funnel-mmd-comparison.json", "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)
