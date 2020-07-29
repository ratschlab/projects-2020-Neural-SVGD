import os
import time
import datetime
import json_tricks as json
import copy

enable_float64 = False
from jax.config import config
config.update("jax_enable_x64", enable_float64)
config.update("jax_debug_nans", True)

from svgd import SVGD
import numpy as onp
import config
import metrics
import utils
import itertools

import colored_traceback.auto

"""
Perform experiment with hyperparameters from config.py
Log metrics, config, and other stuff according to the following structure:

runs/{date-time}/run
runs/{date-time}/config
runs/{date-time}/rundata

containing, respectively,
* text logs with references to other files
* json dumps of hyperparameter config
* json dumps of rundata collected during run (eg loss, metrics)
"""


def run(cfg: dict, svgd: SVGD, logdir: str):
    """Make logfiles in logdir and run experiment.
    Arguments:
    * cfg: entire config (including svgd config) - this will be logged.
    * svgd: instance of SVGD()
    * logdir: directory to log experiment results to."""
    t = time.time()
    startdate = time.strftime("%Y-%m-%d", time.localtime(t))
    starttime = time.strftime("%H:%M:%S", time.localtime(t))
    experiment_name = startdate + "__" + starttime
    logdir = logdir + experiment_name + "/"  # logdir = f"./runs/{experiment_name}/"
    try:
        os.makedirs(logdir)
    except FileExistsError:
        logdir = logdir[:-1] + datetime.datetime.now().strftime(".%f/")
        os.makedirs(logdir)

    files = [logdir + f for f in ["run", "config.json", "rundata.json", "metrics.json"]]
    logfile, configfile, datafile, metricfile = files
    with open(configfile, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4, sort_keys=True)

    print(f"Learning kernel parameters. Writing to {logfile}.")
    kernel_params, log = svgd.train_kernel(**config.get_train_args(cfg))

    et = time.time()
    enddate = time.strftime("%Y-%m-%d", time.localtime(et))
    endtime = time.strftime("%H:%M:%S", time.localtime(et))
    duration= time.strftime("%H:%M:%S", time.gmtime(et - t))

    if not log["Interrupted because of NaN"]:
        metrics_dict = metrics.compute_final_metrics(log["particles"], svgd)
    else:
        metrics_dict = dict()
    # write to logs
    logstring = f"Training started: {startdate} at {starttime}\nTraining ended: {enddate} at {endtime}\nDuration: {duration}\n-----------------\n\nExperiment config and rundata written to:\n{configfile}\n{datafile}"
    with open(logfile, "w") as f:
        f.write(logstring)

    with open(datafile, "w") as f:
        json.dump(utils.tolist(log),          f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)

    with open(metricfile, "w") as f:
        json.dump(utils.tolist(metrics_dict), f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)


def update_config(base_config, run_config, svgd_conf=None, kernel_conf=None, train_conf=None):
    """Modify run_config based on input configurations.
    base_config is never modified.

    Arguments:
    svgd_conf, kernel_conf, train_conf are (not nested) dictionaries."""
    if svgd_conf is not None:
        run_config["svgd"].update(svgd_conf)
    if kernel_conf is not None:
        run_config["kernel"].update(kernel_conf)
        if kernel_conf["architecture"] == "Vanilla":
            run_config["kernel"]["layers"] = []
    if train_conf is not None:
        run_config["train_kernel"].update(train_conf)
        if "svgd_steps" in train_conf:
            if "n_iter" not in train_conf:
                run_config["train_kernel"]["n_iter"] = base_config["train_kernel"]["n_iter"] // train_conf["svgd_steps"]
            if "ksd_steps" not in train_conf:
                run_config["train_kernel"]["ksd_steps"] = train_conf["svgd_steps"]

def grid_search(base_config, hparams, logdir, num_experiments="?"):
    """traverse cartesian product of lists in hparams"""
    os.makedirs(logdir, exist_ok=True)
    starttime = time.strftime("%Y-%m-%d__%H:%M:%S__sweep-config")
    with open(logdir + starttime + ".json", "w") as f:
        json.dump([base_config, hparams], f, ensure_ascii=False, indent=4, sort_keys=True)

    run_config = copy.deepcopy(base_config)
    svgd_configs = utils.dict_cartesian_product(**hparams["svgd"])
    kernel_configs = utils.dict_cartesian_product(**hparams["kernel"])

    counter=1
    for svgd_config, kernel_config in itertools.product(svgd_configs, kernel_configs):
        update_config(base_config, run_config, svgd_conf=svgd_config, kernel_conf=kernel_config)
        svgd = SVGD(**config.get_svgd_args(run_config)) # keep SVGD state, so we don't recompile the kernel every time
        for train_config in utils.dict_cartesian_product(**hparams["train_kernel"]):
            print()
            print(f"Run {counter}/{num_experiments}:")
            update_config(base_config, run_config, train_conf=train_config)
            run(run_config, svgd, logdir)
            counter += 1


if __name__ == "__main__":
    logdir = "./runs/new/"
    num_lr = 5
    d = 2
    k = None

    # hparams
    layers = [
        [4, 4, 2],
    ]
    optimizer_ksd_args = onp.logspace(-2, 1, num=num_lr).reshape((num_lr,1))
    svgd_steps = [1]
    ksd_steps = [1]
    architecture = ["MLP"]

    if k is None:
        target = ["Gaussian"]
    else:
        target=["Gaussian Mixture"]
    onp.random.seed(0)
    target_args=[utils.generate_parameters_for_gaussian(d, k)]
    n_particles = [5000]
    n_subsamples = [200]

    hparams = config.flat_to_nested(dict(layers=layers,
                                         architecture=architecture,
                                         optimizer_ksd_args=optimizer_ksd_args,
                                         svgd_steps=svgd_steps,
                                         ksd_steps=ksd_steps,
                                         target=target,
                                         target_args=target_args,
                                         n_particles=n_particles,
                                         n_subsamples=n_subsamples,
                                         ))

    num_experiments = onp.prod([len(v) for v in utils.flatten_dict(hparams).values()])

    print()
    print("Starting experiment:")
    print(f"Target dimension: {d}")
    if target == ["Gaussian"]:
        print(f"Target shape: Gaussian with parameters:\n* mean {target_args[0][0]}\n* variance {target_args[0][1]}")
    print(f"Float64 enabled: {enable_float64}")
    print(f"Number of modes in mixture: {k if k is not None else 1}")
    print(f"Number of experiments: {num_experiments}")
    print()
    grid_search(config.config, hparams, logdir, num_experiments)
