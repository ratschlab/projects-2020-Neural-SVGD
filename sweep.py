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

logdir
└── {date-time}
    ├── config.json
    ├── metrics.json
    ├── run
    └── rundata.json

containing, respectively,
* json dumps of hyperparameter config
* json dump of final metrics
* text log with references to other files
* json dumps of rundata collected during run (eg loss, metrics)
"""


def run(cfg: dict, logdir: str):
    """Run experiment based on cfg. Log results in logdir.
    Arguments:
    * cfg: entire config (including svgd config) - this will be logged.
    * svgd: instance of SVGD()
    * logdir: directory to log experiment results to."""
    def logfiles():
        t = time.time()
        startdate = time.strftime("%Y-%m-%d", time.localtime(t))
        starttime = time.strftime("%H:%M:%S", time.localtime(t))
        experiment_name = startdate + "__" + starttime
        rundir = logdir + experiment_name + "/"  # logdir = f"./runs/{experiment_name}/"
        try:
            os.makedirs(rundir)
        except FileExistsError:
            rundir = rundir[:-1] + datetime.datetime.now().strftime(".%f/")
            os.makedirs(rundir)
        print(f"Running. Writing to {logdir}.")
        files = [rundir + f for f in ["config.json", "rundata.json", "metrics.json"]]
        return files

    def write_results(rundata: dict, metrics: dict):
        configfile, rundatafile, metricfile = logfiles()
        #et = time.time()
        #enddate = time.strftime("%Y-%m-%d", time.localtime(et))
        #endtime = time.strftime("%H:%M:%S", time.localtime(et))
        #duration= time.strftime("%H:%M:%S", time.gmtime(et - t))
        with open(configfile, "w") as f:
            json.dump(cfg,                   f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)
        with open(rundatafile, "w") as f:
            json.dump(utils.tolist(rundata), f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)
        with open(metricfile, "w") as f:
            json.dump(utils.tolist(metrics), f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)

    svgd = SVGD(**config.get_svgd_args(cfg))
    if cfg["train_kernel"]["train"]:
        _, rundata = svgd.train_kernel(**config.get_train_args(cfg["train_kernel"]))
    else:
        rundata = svgd.sample(**config.get_sample_args(cfg["train_kernel"]))
    if not rundata["Interrupted because of NaN"]:
        metrics_dict = metrics.compute_final_metrics(rundata["particles"], svgd)
    else:
        metrics_dict = dict()
    write_results(rundata, metrics_dict)


def make_config(base_config: dict, run_options: dict):
    """
    Arguments:
    * base_config: complete base config (eg the one in config.py)
    * run_options: same tree structure as base_config, but potentially missing entries

    Returns:
    * run_config: dict with same (complete) tree structure as base_config.
    Entries are populated based on the following logic:
    1) use entry in run_options if it exists
    2) else maybe dynamically populate the entry, based on other entries in run_options and base_config
    3) else use entry from base_config
    """
    base_config = utils.flatten_dict(base_config)
    run_options = utils.flatten_dict(run_options)
    def infer_value(key):
        """Use if key not in run_options. Dynamically infers value, or
        else just returns value from base_config"""
        # for a vanilla (non kernel learning) run, set useless configs to null
        if "train" in run_options:
            if not run_options["train"] and key in [
                    "encoder_layers",
                    "decoder_layers",
                    "ksd_steps",
                    "optimizer_ksd",
                    "optimizer_ksd_args",
                    "lambda_reg",
            ]:
                return None
        # adapt number iterations / ksd steps based on svgd steps per iter
        if "svgd_steps" in run_options:
            if key == "n_iter":
                return base_config["n_iter"] // run_options["svgd_steps"]
            if key == "ksd_steps":
                return run_options["svgd_steps"]
        if "encoder_layers" in run_options:
            if key == "decoder_layers":
                ls = copy.copy(run_options["encoder_layers"])
                ls.reverse()
                ls[-1] = len(run_options["target_args"][0])
                return ls
        return base_config[key]

    run_config = dict()
    for k, v in base_config.items():
        if k in run_options:
            run_config[k] = run_options[k]
        else:
            run_config[k] = infer_value(k)
    return config.flat_to_nested(run_config)

def grid_search(base_config, sweep_config, logdir, num_experiments="?"):
    """traverse cartesian product of lists in sweep_config"""
    os.makedirs(logdir, exist_ok=True)
    starttime = time.strftime("%Y-%m-%d__%H:%M:%S__sweep-config")
    with open(logdir + starttime + ".json", "w") as f:
        json.dump([base_config, sweep_config], f, ensure_ascii=False, indent=4, sort_keys=True)

    svgd_configs  = utils.dict_cartesian_product(**sweep_config["svgd"])
    train_configs = utils.dict_cartesian_product(**sweep_config["train_kernel"])
    counter=1
    for svgd_config, train_config in itertools.product(svgd_configs, train_configs):
        print()
        print(f"Run {counter}/{num_experiments}:")
        run_options = {"svgd": svgd_config, "train_kernel": train_config}
        run_config = make_config(base_config, run_options)
        run(run_config, logdir)
        counter += 1

if __name__ == "__main__":
    logdir = "./runs/two-dim/"
    num_lr = 7
    d = 2
    k = None
    if k is None:
        target = ["Gaussian"]
    else:
        target=["Gaussian Mixture"]

    # sweep_config
    encoder_layers = [
        [4, 4, 2],
        [16, 16, 2],
        [16, 16, 16, 2],
    ]

    optimizer_ksd_args = onp.logspace(-2, 1, num=num_lr).reshape((num_lr,1))
    lambda_reg = onp.logspace(-2, 3, num=num_lr).reshape((num_lr,1))
    svgd_steps = [1]
    ksd_steps = [1, 2, 5]
    n_iter = [50]

    onp.random.seed(0)
    target_args=[utils.generate_parameters_for_gaussian(d, k)]
    n_particles = [5000]
    n_subsamples = [200]

    sweep_config = config.flat_to_nested(dict(
        train=[True],
        optimizer_ksd_args=optimizer_ksd_args,
        encoder_layers=encoder_layers,
        svgd_steps=svgd_steps,
        ksd_steps=ksd_steps,
        target=target,
        target_args=target_args,
        n_particles=n_particles,
        n_subsamples=n_subsamples,
        n_iter=n_iter,
    ))

    num_experiments = onp.prod([len(v) for v in utils.flatten_dict(sweep_config).values()])

    print()
    print("Starting experiment:")
    print(f"Target dimension: {d}")
    if target == ["Gaussian"]:
        print(f"Target shape: Gaussian with parameters:\n* mean {target_args[0][0]}\n* variance {target_args[0][1]}")
    print(f"Float64 enabled: {enable_float64}")
    print(f"Number of modes in mixture: {k if k is not None else 1}")
    print(f"Number of experiments: {num_experiments}")
    print()
    #grid_search(config.config, sweep_config, logdir, num_experiments)

    # vanilla runs
    vanilla_config = config.flat_to_nested(dict(
        train=[False],
        svgd_steps=svgd_steps,
        target=target,
        target_args=target_args,
        n_particles=n_particles,
        n_subsamples=n_subsamples,
        n_iter=n_iter,
    ))
    num_experiments = onp.prod([len(v) for v in utils.flatten_dict(vanilla_config).values()])
    print("Starting vanilla runs:")
    print(f"Number of runs: {num_experiments}")
    print()
    grid_search(config.config, vanilla_config, logdir, num_experiments)
