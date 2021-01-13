import jax.numpy as jnp
from jax import jit, random 
import matplotlib
import numpy as onp


import distributions
import flows
import kernels
import metrics

import config as cfg


from tqdm import tqdm
key = random.PRNGKey(0)


# Config
# set up exporting
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'pgf.rcfonts': False,
    'axes.unicode_minus': False, # avoid unicode error on saving plots with negative numbers (??)
})

#figure_path = "/home/lauro/documents/msc-thesis/paper/latex/figures/"
figure_path = "../results/figures/"
results_path = "../results/"

NUM_STEPS = 500
PARTICLE_STEP_SIZE = 1e-2 # for particle update
LEARNING_RATE = 1e-4 # for neural network
N_PARTICLES = 100 # 200
MAX_DIM = 4 # sweep from 2 to MAX_DIM


funnel = kernels.get_funnel_kernel(1.)
mmd = jit(metrics.get_mmd(funnel))


def get_mmds(particle_list, ys):
    mmds = []
    for xs in [p.particles.training for p in particle_list]:
        mmds.append(mmd(xs, ys))
    return mmds

def sample(d, key, n_particles):
    target = distributions.Funnel(d)
    proposal = distributions.Gaussian(np.zeros(d), np.ones(d))
    funnel_setup = distributions.Setup(target, proposal)

    key, subkey = random.split(key)
    neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, funnel_setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., patience=0, learner_lr=learner_lr)
    svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=0., scaled=True,  bandwidth=None)
    sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=n_steps, particle_lr=particle_lr, noise_level=1.)
    
    return (neural_particles, svgd_particles, sgld_particles), (neural_learner, svgd_gradient, sgld_gradient)


mmd_sweep = []
for d in tqdm(range(2, MAX_DIM), disable=True):
    print(d)
    key, subkey = random.split(key)
    particles, gradients = sample(d, subkey, n_particles)

    target = distributions.Funnel(d)
    key, subkey = random.split(key)
    ys = target.sample(n_particles, subkey)
    mmds = get_mmds(particles, ys)
    mmd_sweep.append(mmds)
mmd_sweep = onp.array(mmd_sweep)

#plt.subplots(figsize=[23, 8])
#names = "Neural SVGD SGLD".split()
#lines = plt.plot(mmd_sweep, "--.")
#for name, line in zip(names, lines):
#    line.set_label(name)
#    
#plt.ylabel("MMD(samples, target)")
#plt.xlabel("Dimensionality of sample space")
#plt.legend()
#plt.yscale("log")


# save json results
results = {
        "NSVGD": mmd_sweep[:, 0].tolist(),
        "SVGD":  mmd_sweep[:, 1].tolist(),
        "SGLD":  mmd_sweep[:, 2].tolist(),
}

with open(results_path + "funnel-dimension-sweep.json", "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)
