import os
import utils
import config as cfg
import numpy as onp
from bayes_nsvgd import train
from jax import random


OVERWRITE_FILE = False

key = random.PRNGKey(0)
key, subkey = random.split(key)
EVALUATE_EVERY = 25
results_file = cfg.results_path + "bnn-sweep.csv"

sweep_dict = {
    "meta_lr": onp.logspace(start=-5, stop=-2, num=4),
    "particle_stepsize": onp.logspace(start=-5, stop=-2, num=4),
    "patience": [0],
    "max_train_steps": [10, 50],
}


def get_csv_string(params: dict):
    """convert dict to string of comma-separated values"""
    return ",".join(str(v) for v in params.values())


if not os.path.isfile(results_file) or OVERWRITE_FILE:
    with open(results_file, "w") as file:
        file.write(",".join(sweep_dict.keys()) + ",step,accuracy\n")

key, subkey = random.split(key)
for param_setting in utils.dict_cartesian_product(**sweep_dict):
    steps, accuracies = train(key=subkey,
                              dropout=True,
                              evaluate_every=EVALUATE_EVERY,
                              **param_setting)
    param_string = get_csv_string(param_setting)
    with open(results_file, "a") as file:
        for step, accuracy in zip(steps, accuracies):
            file.write(param_string + f"{step},{accuracy}\n")