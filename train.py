import os
import time
import json
import copy

from svgd import SVGD
import numpy as onp
import config

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

def run(cfg, logdir):
    t = time.time()
    startdate = time.strftime("%Y-%m-%d", time.localtime(t))
    starttime = time.strftime("%H:%M:%S", time.localtime(t))
    experiment_name = startdate + "__" + starttime
    logdir = logdir + experiment_name + "/"  # logdir = f"./runs/{experiment_name}/"
    logfile, configfile, datafile = [logdir + f for f in ["run", "config", "data"]]
    for f in [logfile, configfile, datafile]:
        os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(configfile, "w") as f:
        f.write(json.dumps(cfg))

    svgd = SVGD(**config.get_svgd_args(cfg))
    print("Learning kernel parameters...")
    kernel_params, log = svgd.train_kernel(**config.get_train_args(cfg))

    et = time.time()
    enddate = time.strftime("%Y_%m_%d", time.localtime(et))
    endtime = time.strftime("%H:%M:%S", time.localtime(et))
    duration= time.strftime("%H:%M:%S", time.gmtime(et - t))

    # write to logs
    logstring = f"Training started: {startdate} at {starttime}\nTraining ended: {enddate} at {endtime}\nDuration: {duration}\n-----------------\n\nExperiment config and data written to:\n{configfile}\n{datafile}"
    with open(logfile, "w") as f:
        f.write(logstring)

    with open(datafile, "w") as f:
        f.write(json.dumps({k: onp.asarray(v).tolist() for k, v in log.items()}))


if __name__ == "__main__":
    logdir = "./runs/test1/"
    layers = [
        [32, 2],
        [32, 32, 2],
        [32, 32, 32, 2],
        [32, 32],
        [32, 32, 32],
    ]

    architecture = [
        "MLP",
    ]

    optimizer_svgd_args = [[5], [1], [0.1]]
#    optmizer_svgd = ["Adagrad", "Adam"]
    base_config = config.config

    for lay in layers:
        for opt_s in optimizer_svgd_args:

            run_config = copy.deepcopy(base_config)
            run_config["kernel"]["layers"] = lay
            run_config["svgd"]["optimizer_svgd_args"] = opt_s

            run(run_config, logdir)
