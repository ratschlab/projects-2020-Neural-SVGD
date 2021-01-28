import os
import config as cfg
import numpy as onp
import nvgd_bnn
import svgd_bnn
import sgld_bnn
from jax import random

DEBUG = False
if DEBUG:
    NUM_STEPS = 2
    n_lrs = 1
else:
    NUM_STEPS = 200
    n_lrs = 5

OVERWRITE_FILE = True
EVALUATE_EVERY = -1  # never

key = random.PRNGKey(0)
key, subkey = random.split(key)
results_file = "/dev/null"
sweep_results_file = cfg.results_path + "sweep.csv"  # best LR / acc goes here
temp_results_file = cfg.results_path + "temp_sweep.csv"  # results from entire sweep go here

vgd_stepsizes = onp.logspace(start=-7, stop=-3, num=n_lrs)
sgld_stepsizes = onp.logspace(start=-9, stop=-5, num=n_lrs)

if not os.path.isfile(sweep_results_file) or OVERWRITE_FILE:
    with open(sweep_results_file, "w") as f:
        f.write("name,optimal_stepsize,max_val_accuracy\n")


with open(temp_results_file, "w") as f:
    f.write("name,stepsize,val_accuracy\n")


def save_single_run(name, accuracy, step_size):
    with open(temp_results_file, "a") as f:
        f.write(f"{name},{step_size},{accuracy}\n")


def get_best_run(name, accuracy_list):
    """
    return best step size and highest accuracy
    args:
        name: name of sweep to save in csv file
        accuracy_list: list with entries (accuracy, step_size)
    """
    accuracy_list = onp.array(accuracy_list)
    argmax_idx = accuracy_list[:, 0].argmax()
    max_accuracy, max_stepsize = accuracy_list[argmax_idx].tolist()

    print(f"Max accuracy {max_accuracy} achieved using step size {max_stepsize}.")
    print()

    with open(sweep_results_file, "a") as f:
        f.write(f"{name},{max_stepsize},{max_accuracy}\n")

    return max_accuracy, max_stepsize


print("Sweeping NVGD...")
final_accs = []
key, subkey = random.split(key)
for particle_stepsize in vgd_stepsizes:
    final_acc = nvgd_bnn.train(key=subkey,
                               particle_stepsize=particle_stepsize,
                               n_iter=NUM_STEPS,
                               evaluate_every=EVALUATE_EVERY,
                               overwrite_file=OVERWRITE_FILE,
                               dropout=True,
                               results_file=results_file)
    save_single_run("nvgd", final_acc, particle_stepsize)
    final_accs.append((final_acc, particle_stepsize))

max_accuracy, nvgd_max_stepsize = get_best_run("nvgd", final_accs)


print("Sweeping Langevin...")
for particle_stepsize in sgld_stepsizes:
    final_acc = sgld_bnn.train(key=subkey,
                               particle_stepsize=particle_stepsize,
                               n_iter=NUM_STEPS,
                               evaluate_every=EVALUATE_EVERY,
                               results_file=results_file)
    final_accs.append((final_acc, particle_stepsize))
    save_single_run("sgld", final_acc, particle_stepsize)

max_accuracy, sgld_max_stepsize = get_best_run("sgld", final_accs)


print("Sweeping SVGD...")
for particle_stepsize in vgd_stepsizes:
    final_acc = svgd_bnn.train(key=subkey,
                               particle_stepsize=particle_stepsize,
                               n_iter=NUM_STEPS,
                               evaluate_every=EVALUATE_EVERY,
                               results_file=results_file)
    final_accs.append((final_acc, particle_stepsize))
    save_single_run("svgd", final_acc, particle_stepsize)

max_accuracy, svgd_max_stepsize = get_best_run("svgd", final_accs)