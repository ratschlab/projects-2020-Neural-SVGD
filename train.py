import os
import time
import json
import copy

from svgd import SVGD
import numpy as onp
import config
import metrics
import utils
import itertools

"""
Perform experiment with hyperparameters from config.py
Log metrics, config, and other stuff according to the following structure:

runs/{date-time}/run
runs/{date-time}/config
runs/{date-time}/data

containing, respectively,
* text logs with references to other files
* json dumps of hyperparameter config
* json dumps of data collected during run (eg loss, metrics)
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
    os.makedirs(logdir, exist_ok=True)
    files = [logdir + f for f in ["run", "config", "data", "metrics"]]
    if os.path.exists(files[0]):
        files = [f + "-new" for f in files]
    logfile, configfile, datafile, metricfile = files
    with open(configfile, "w") as f:
        f.write(json.dumps(cfg))

    print(f"Learning kernel parameters. Writing to {logfile}.")
    kernel_params, log = svgd.train_kernel(**config.get_train_args(cfg))

    et = time.time()
    enddate = time.strftime("%Y-%m-%d", time.localtime(et))
    endtime = time.strftime("%H:%M:%S", time.localtime(et))
    duration= time.strftime("%H:%M:%S", time.gmtime(et - t))

    metric: str = str(metrics.compute_final_metric(log["particles"], svgd))
    # write to logs
    logstring = f"Training started: {startdate} at {starttime}\nTraining ended: {enddate} at {endtime}\nDuration: {duration}\n-----------------\n\nExperiment config and data written to:\n{configfile}\n{datafile}"
    with open(logfile, "w") as f:
        f.write(logstring)

    with open(datafile, "w") as f:
        f.write(json.dumps({k: onp.asarray(v).tolist() for k, v in log.items()}))

    with open(metricfile, "w") as f:
        f.write(metric)


def grid_search(base_config, hparams, logdir):
    """traverse cartesian product of lists in hparams"""
    os.makedirs(logdir, exist_ok=True)
    starttime = time.strftime("%Y-%m-%d__%H:%M:%S__sweep-config")
    with open(logdir + starttime, "w") as f:
        f.write(json.dumps([base_config, hparams]))

    run_config = copy.deepcopy(base_config)
    svgd_configs = utils.dict_cartesian_product(**hparams["svgd"])
    kernel_configs = utils.dict_cartesian_product(**hparams["kernel"])

    for svgd_config, kernel_config in itertools.product(svgd_configs, kernel_configs):
        run_config["svgd"].update(svgd_config)
        run_config["kernel"].update(kernel_config)
        svgd = SVGD(**config.get_svgd_args(run_config)) # keep SVGD state, so we don't recompile the kernel every time
        for train_config in utils.dict_cartesian_product(**hparams["train_kernel"]):
            run_config["train_kernel"].update(train_config)
            run(run_config, svgd, logdir)


if __name__ == "__main__":
    logdir = "./runs/vanilla/"
    layers = [ # TODO: hparams["kernel"] is ignored by grid_search
        [32, 32],
    ]
    optimizer_svgd_args = [[0.5], [1], [2], [5]]
    svgd_steps = [1]
    ksd_steps = [1, 2, 3, 5, 10]
    n_iter = [100]
    architecture = ["Vanilla"]


    hparams = config.flat_to_nested(dict(
                                         architecture=architecture,
                                         optimizer_svgd_args=optimizer_svgd_args,
                                         ksd_steps=ksd_steps,
                                         svgd_steps=svgd_steps,
                                         n_iter=n_iter))
    grid_search(config.config, hparams, logdir)
