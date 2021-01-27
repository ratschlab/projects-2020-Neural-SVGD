import utils
import config as cfg
import numpy as onp
from bayes_nsvgd import train
from jax import random

OVERWRITE_FILE = False
NUM_STEPS = 201
EVALUATE_EVERY = 100

key = random.PRNGKey(0)
key, subkey = random.split(key)
results_file = cfg.results_path + "sweep-bnn.csv"

sweep_dict = {
    "meta_lr": onp.logspace(start=-3, stop=-2, num=4),
    "particle_stepsize": onp.logspace(start=-4, stop=-1, num=4),
    "patience": [0, 5],
    "max_train_steps_per_iter": [10, 50],
    "particle_steps_per_iter": [1],
}

key, subkey = random.split(key)
for param_setting in utils.dict_cartesian_product(**sweep_dict):
    train(key=subkey,
          n_iter=NUM_STEPS,
          evaluate_every=EVALUATE_EVERY,
          overwrite_file=OVERWRITE_FILE,
          dropout=True,
          results_file=results_file,
          **param_setting)
