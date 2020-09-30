import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning")
import json_tricks as json
import warnings

import matplotlib as mpl

import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd
from jax import lax
from jax.ops import index_update, index
import matplotlib.pyplot as plt
import numpy as onp
import jax
import pandas as pd
import haiku as hk

import config

import utils
import metrics
import time
import plot
import svgd
import stein
import train
import kernels
from svgd import SVGD

from jax.experimental import optimizers

rkey = random.PRNGKey(0)
basedir = "/home/lauro/code/msc-thesis/svgd/"


def get_data(rundir):
    """Take directory with run results and return a Pandas DataFrame with the relevant hparams plus metrics.
    Returns:
    * frame: a pandas dataframe with config hparams and metrics
    * data: a list of dicts with data collected during runs"""
    cfgs = []
    rundata = []
    metrics = []
    sweep_metaconfigs = []
    base_metaconfigs = []
    for logdir in os.listdir(rundir):
        if os.path.isdir(os.path.join(rundir, logdir)):
            configfile, datafile, metricfile = [rundir + logdir + f for f in ["/config.json", "/data.json", "/metrics.json"]]
            with open(configfile, "r") as f:
                cfgs.append(json.load(f))
            try:
                with open(datafile, "r") as f:
                    rundata.append(json.load(f))
                with open(metricfile, "r") as f:
                    metrics.append(json.load(f))
            except FileNotFoundError:
                warnings.warn(f"No config / data files in {logdir}.")
        else:
            with open(rundir + logdir, "r") as f:
                base, sweep_config = json.load(f)
                sweep_metaconfigs.append(sweep_config)
                base_metaconfigs.append(base)
     
    # process data
    cfgs_flat = [utils.flatten_dict(c) for c in cfgs]
    for c in cfgs_flat:
        if len(c["optimizer_svgd_args"]) == 1:
            c["optimizer_svgd_args"] = onp.squeeze(c["optimizer_svgd_args"])
        if len(c["optimizer_ksd_args"]) == 1:
            c["optimizer_ksd_args"] = onp.squeeze(c["optimizer_ksd_args"])
    
    
    configs_df = pd.DataFrame(cfgs_flat)
    configs_df.rename(columns={"optimizer_svgd_args": "lr svgd", "optimizer_ksd_args": "lr ksd"}, inplace=True)
    
    metrics_df = pd.DataFrame(metrics)
    all_df = pd.concat([metrics_df, configs_df], axis=1)
    all_df["layers"] = all_df.layers.astype('str').astype('category')
    all_df["architecture"] = all_df.architecture.astype('str').astype('category')
    
    if all_df.isnull().values.any():
        rows_with_nans = all_df.shape[0] - all_df.dropna().shape[0]
        warnings.warn(f"Detected NaNs in dataframe. {rows_with_nans} / {all_df.shape[0]} rows include at least one NaN. Dropping all rows with NaNs.")
        all_df = all_df.dropna()

    return all_df, rundata, cfgs, sweep_metaconfigs, base_metaconfigs


def plot_pdf(cfg: dict, log: dict):
    p = SVGD(**config.get_svgd_args(cfg)).target
    particles = np.array(log["particles"])[:, 0]
    lims = onp.squeeze([p.mean + d for d in (-1.5 * p.cov, 1.5 * p.cov)])
    
    grid = np.linspace(*lims, 100)
    plt.plot(grid, vmap(p.pdf)(grid))
    plt.hist(particles, density=True)


ls ../runs


rundir = basedir + "runs/four-dim/"
frame, rundata, configs, sweep_metaconfigs, base_metaconfigs = get_data(rundir)
relevant = ["ksd", "emd", "sinkhorn_divergence"] + ["lr ksd", "svgd_steps", "architecture", "layers"]
f = frame[relevant]
# f


f


fig, ax = plt.subplots(figsize=[7, 8])
f_mlp = f[f.architecture == "MLP"]

f_mlp = f_mlp[f_mlp["emd"] < 2.0]
plt.scatter(f_mlp["layers"], f_mlp["emd"], c=f_mlp["lr ksd"], cmap="inferno", norm=mpl.colors.LogNorm(), s=100)
plt.yscale("log")
plt.colorbar()


f_mlp = f[f.architecture == "MLP"]
plt.scatter(f_mlp["lr ksd"], f_mlp.emd)


plt.scatter(f_mlp["svgd_steps"], f_mlp.emd)


f_v = f[f.architecture == "Vanilla"]
plt.scatter(f_v["svgd_steps"], f_v["sinkhorn_divergence"])
plt.yscale("log")











sldkfjsl


f


plt.plot(f["architecture"], f["emd"], "o")
plt.yscale("log")


plt.plot(f["n_iter"], f["svgd_steps"], "o")


pd.Categorical(f.architecture)


f_mlp = f[f["architecture"] == "MLP"]
plt.plot(f_mlp["lr svgd"], f_mlp["emd"], "o")
plt.xscale("log")


f_mlp.plot.scatter("lr svgd", "emd", c="svgd_steps", cmap="viridis")
plt.xscale("log")


n = 0
fig, axs = plt.subplots(1, 3, figsize=[20,4])
axs[0].plot(data[n]["mean"])
axs[1].plot(data[n]["var"])
plot_pdf(configs[n], data[n])

