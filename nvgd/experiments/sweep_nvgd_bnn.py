import utils
import config as cfg
import numpy as onp
import json_tricks as json
import nvgd_bnn
from jax import random

OVERWRITE_FILE = True
NUM_STEPS = 200
EVALUATE_EVERY = 100

key = random.PRNGKey(0)
key, subkey = random.split(key)
results_file = cfg.results_path + "sweep-nvgd-bnn.csv"

sweep_dict = {
    "meta_lr": onp.logspace(start=-4, stop=-2, num=3),
    "particle_stepsize": onp.logspace(start=-4, stop=0, num=6),
    "patience": [0, 5],
    "max_train_steps_per_iter": [10, 50],
    "particle_steps_per_iter": [1],
}

final_accs = []
key, subkey = random.split(key)
for param_setting in utils.dict_cartesian_product(**sweep_dict):
    final_acc = nvgd_bnn.train(key=subkey,
                               n_iter=NUM_STEPS,
                               evaluate_every=EVALUATE_EVERY,
                               overwrite_file=OVERWRITE_FILE,
                               dropout=True,
                               results_file=results_file,
                               **param_setting)
    final_accs.append((final_acc, param_setting))

argmax_idx = onp.array([a[0] for a in final_accs]).argmax()
max_accuracy = final_accs[argmax_idx][0]
max_psetting = final_accs[argmax_idx][1]
print(f"Max accuracy {max_accuracy} achieved using setting")
print(json.dumps(max_psetting, indent=4))