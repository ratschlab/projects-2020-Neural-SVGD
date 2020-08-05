# coding: utf-8
import os
import time
import datetime
import json_tricks as json
import copy
import argparse

on_cluster = not os.getenv("HOME") == "/home/lauro"
if on_cluster:
    test_logging = False
    debug_nans = False
else:
    test_logging = False
    debug_nans = False
enable_float64 = False
from jax.config import config
config.update("jax_enable_x64", enable_float64)
config.update("jax_debug_nans", debug_nans)
from jax import random

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
parser = argparse.ArgumentParser()
parser.add_argument("--key", type=int, default=0, help="Random seed")
parser.add_argument("--target", type=str, default="", help="Name of target."
                    "Must be either 'banana' or the emtpy string.")
parser.add_argument("--dim", type=int, default=2, help="Dimension of target."
                    "Only needed when --target='', otherwise it is ignored.")
args = parser.parse_args()

def run(key, cfg: dict, logdir: str):
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
        print(f"Writing results to {logdir}.")
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
            json.dump(utils.dict_dejaxify(rundata), f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)
        with open(metricfile, "w") as f:
            json.dump(utils.dict_dejaxify(metrics), f, ensure_ascii=False, indent=4, sort_keys=True, allow_nan=True)

    svgd = SVGD(**config.get_svgd_args(cfg))
    if test_logging:
        rundata = {}
        metrics_dict = {}
    else:
        if cfg["train_kernel"]["train"]:
            _, rundata = svgd.train_kernel(key, **config.get_train_args(cfg["train_kernel"]))
        else:
            rundata = svgd.sample(key, **config.get_sample_args(cfg["train_kernel"]))
        if not rundata["Interrupted because of NaN"]:
            metrics_dict = metrics.compute_final_metrics(rundata["particles"][rundata["validation_idx"]], svgd)
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
                    "lr_ksd",
                    "lambda_reg",
                    "svgd_steps",
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

def random_search(key, base_config: dict, sweep_config: dict, hparams: list,
                  logdir: str, n_random_samples: int):
    """
    Arguments:
    * key: PRNGKey
    * base_config: complete base config (eg the one in config.py)
    * sweep_config: same tree structure as base_config, but potentially missing entries.
      All entries are lists of values.
    * hparams: list of strings, e.g. ["lr_ksd", "lambda_reg"]
    * logdir: where to log runs
    * n_random_samples: number of randomly sampled hparams per sweep config

    1) Iterate through values in sweep_config
    2) For each config, sample hparams randomly and run. repeat this x times

    config parameters:
    * encoder_layers
    * decoder_layers
    * ksd_steps
    * target_args
    * ...

    hparams:
    * lr_ksd
    * lr_svgd
    * lambda_reg
    """
    num_experiments = onp.prod([len(v) for v in utils.flatten_dict(sweep_config).values()])
    num_experiments *= n_random_samples

    os.makedirs(logdir, exist_ok=True)
    starttime = time.strftime("%Y-%m-%d__%H:%M:%S__config")
    with open(logdir + starttime + "-base.json", "w") as f:
        json.dump(base_config, f, ensure_ascii=False,
                  indent=4, sort_keys=True)
    with open(logdir + starttime + "-sweep.json", "w") as f:
        json.dump(sweep_config, f, ensure_ascii=False,
                  indent=4, sort_keys=True)

    svgd_configs  = utils.dict_cartesian_product(**sweep_config["svgd"])
    train_configs = utils.dict_cartesian_product(**sweep_config["train_kernel"])
    product_config = list(itertools.product(svgd_configs, train_configs))
    counter=1
    key, skey = random.split(key)
    for subkey in random.split(skey, n_random_samples):
        for svgd_config, train_config in product_config:
            print()
            print(f"Run {counter}/{num_experiments}")
            run_options = {}
            run_options.update(svgd_config)
            run_options.update(train_config)
            run_options.update(sample_hparams(subkey, *hparams))
            run_options = config.flat_to_nested(run_options)
            key, skey = random.split(key)
            run_config = make_config(base_config, run_options)
            run(skey, run_config, logdir)
            counter += 1

def sample_hparams(key, *names):
    keys = random.split(key, len(names))
    samplers = {
        "lr_ksd":     lambda key: 10**random.uniform(key, minval=-3, maxval=1),
        "lr_svgd":    lambda key: 10**random.uniform(key, minval=-2, maxval=1.5),
        "lambda_reg": utils.mixture(
            [lambda key: 10**random.uniform(key, minval=-5, maxval=2),
             lambda key: 0],
            [6/7, 1/7]
        )
    }
    return {name: float(samplers[name](key)) for name, key in zip(names, keys)}


if __name__ == "__main__":
    n_iter = [70]
    d = args.dim
    key = random.PRNGKey(args.key)
    target_name = args.target # one of '', 'banana'. If empty string, then target
                              # mean and var are chosen randomly.
    if target_name=="banana": d=2
    logdir = f"./runs/{d}-dim/" if target_name=="" else f"./runs/{d}-dim-{target_name}/"
    if target_name=="":
        k = None
        if k is None:
            target = ["Gaussian"]
        else:
            target=["Gaussian Mixture"]
        onp.random.seed(42)
        target_args=[utils.generate_parameters_for_gaussian(d, k)]
    elif target_name=="banana":
        target = ["Gaussian Mixture"]
        target_args = [metrics.bent_args]
    else:
        raise ValueError("target name must be either 'banana' or the empty string."
                         f"Instead received: {target_name}.")
    # sweep_config
    encoder_layers = [
        [4, 4, 2],
        [4, 4, 1],
        [16, 16, 32, 16, 2],
    ]

    svgd_steps = [1]
    ksd_steps = [5, 10]

    n_particles = [6000]
    n_subsamples = [100] # recall subsamples for ksd are 20x this
    minimize_ksd_variance = [True, False]

    sweep_config = config.flat_to_nested(dict(
        train=[True],
        encoder_layers=encoder_layers,
        svgd_steps=svgd_steps,
        ksd_steps=ksd_steps,
        target=target,
        target_args=target_args,
        n_particles=n_particles,
        n_subsamples=n_subsamples,
        n_iter=n_iter,
        minimize_ksd_variance=minimize_ksd_variance,
    ))

    vanilla_config = config.flat_to_nested(dict(
        train=[False],
        svgd_steps=svgd_steps,
        target=target,
        target_args=target_args,
        n_particles=n_particles,
        n_subsamples=n_subsamples,
        n_iter=n_iter,
    ))

    hparams = ["lr_ksd", "lambda_reg", "lr_svgd"]
    vanilla_hparams = ["lr_svgd"]
    n_random_samples = 5
    key, subkey = random.split(key)

    # vanilla runs
    key, subkey = random.split(key)
    n_random_samples_vanilla = 3

    print("Starting experiments.")
    print(f"Target dimension: {d}")
    if target == ["Gaussian"]:
        print(f"Target shape: Gaussian with parameters:\n* mean {target_args[0][0]}\n* variance {target_args[0][1]}")
    print(f"Float64 enabled: {enable_float64}")
    print()
    random_search(subkey, config.config, vanilla_config, vanilla_hparams, logdir, n_random_samples_vanilla)
    random_search(subkey, config.config, sweep_config,   hparams,         logdir, n_random_samples)
