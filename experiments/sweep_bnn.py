import os
import argparse
import config as cfg
import numpy as onp
import nvgd_bnn
import svgd_bnn
import sgld_bnn
from jax import random

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default='all',
                    help="Which method to sweep. Can be 'nvgd', 'sgld',"
                         "'svgd', or 'all'.")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()


DEBUG = args.debug
if DEBUG:
    print("Running in debug mode")
    NUM_STEPS = 2
    n_lrs = 1
else:
    NUM_STEPS = 200
    n_lrs = 5

OVERWRITE_FILE = True
EVALUATE_EVERY = -1  # never

key = random.PRNGKey(0)
key, subkey = random.split(key)
results_path = cfg.results_path + "bnn-sweep/"
sweep_results_file = results_path + "best-stepsizes.csv"  # best LR / acc goes here
dumpfile = "/dev/null"
final_accs = []

vgd_stepsizes = onp.logspace(start=-7, stop=-3, num=n_lrs)
sgld_stepsizes = onp.logspace(start=-9, stop=-5, num=n_lrs)

if not os.path.isfile(sweep_results_file) or OVERWRITE_FILE:
    with open(sweep_results_file, "w") as f:
        f.write("name,optimal_stepsize,max_val_accuracy\n")


def save_single_run(name, accuracy, step_size):
    file = results_path + name + ".csv"
    if not os.path.isfile(file):
        with open(file, "w") as f:
            f.write("stepsize,accuracy\n")

    with open(file, "a") as f:
        f.write(f"{step_size},{accuracy}\n")


def save_best_run(name, accuracy_list):
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


def sweep_nvgd():
    print("Sweeping NVGD...")
    for particle_stepsize in vgd_stepsizes:
        final_acc = nvgd_bnn.train(key=subkey,
                                   particle_stepsize=particle_stepsize,
                                   n_iter=NUM_STEPS,
                                   evaluate_every=EVALUATE_EVERY,
                                   overwrite_file=OVERWRITE_FILE,
                                   dropout=True,
                                   results_file=dumpfile)
        save_single_run("nvgd", final_acc, particle_stepsize)
        final_accs.append((final_acc, particle_stepsize))

    max_accuracy, nvgd_max_stepsize = save_best_run("nvgd", final_accs)


def sweep_sgld():
    print("Sweeping Langevin...")
    for particle_stepsize in sgld_stepsizes:
        final_acc = sgld_bnn.train(key=subkey,
                                   particle_stepsize=particle_stepsize,
                                   n_iter=NUM_STEPS,
                                   evaluate_every=EVALUATE_EVERY,
                                   results_file=dumpfile)
        final_accs.append((final_acc, particle_stepsize))
        save_single_run("sgld", final_acc, particle_stepsize)

    max_accuracy, sgld_max_stepsize = save_best_run("sgld", final_accs)


def sweep_svgd():
    print("Sweeping SVGD...")
    for particle_stepsize in vgd_stepsizes:
        final_acc = svgd_bnn.train(key=subkey,
                                   particle_stepsize=particle_stepsize,
                                   n_iter=NUM_STEPS,
                                   evaluate_every=EVALUATE_EVERY,
                                   results_file=dumpfile)
        final_accs.append((final_acc, particle_stepsize))
        save_single_run("svgd", final_acc, particle_stepsize)

    max_accuracy, svgd_max_stepsize = save_best_run("svgd", final_accs)


if args.run == "nvgd":
    sweep_nvgd()
elif args.run == "sgld":
    sweep_sgld()
elif args.run == "svgd":
    sweep_svgd()
elif args.run == "all":
    sweep_sgld()
    sweep_svgd()
    sweep_nvgd()
else:
    raise ValueError("cli argument 'run' must be one of 'nvgd', 'sgld',"
                     "'svgd', or 'all'.")
