import haiku as hk
from jax import random
from jax.experimental import optimizers
import metrics
import kernels
import svgd
import utils

config = dict()
config["svgd"] = {
    "target": "Gaussian",  # one of ["Gaussian", "Gaussian Mixture"]
    "target_args": [[-2, 3], [1, 6]],  # either [mean, cov] or [means, covs, weights]
    "n_particles": 1000,
    "n_subsamples": 300,
    "optimizer_svgd": "Adagrad",  # One of ["Adam", "Adagrad", "SGD"]
    "optimizer_svgd_args": [1.0],
    "subsample_with_replacement": False,
    "lam": 1.0,
}

config["kernel"] = {
    "encoder_layers": [4, 4, 2],
    "decoder_layers": [4, 4, 2],
    "kernel": "ard", # no alternatives atm
}

config["train_kernel"] = {
    "key": 0,
    "n_iter": 50,
    "ksd_steps": 1,
    "svgd_steps": 1,
    "optimizer_ksd": "Adam",  # One of ["Adam", "Adagrad", "SGD"]. optimizer for encoder and decoder
    "optimizer_ksd_args": [0.03]
}

####### utilities
opts = {
    "Adam": optimizers.adam,
    "Adagrad": optimizers.adagrad,
    "SGD": optimizers.sgd,
    "RMSProp": optimizers.rmsprop,
}

def get_svgd_args(config):
    targets = {
        "Gaussian": metrics.Gaussian,
        "Gaussian Mixture": metrics.GaussianMixture
    }

    cfg = config["svgd"]
    kcfg = config["kernel"]

    encoder = hk.transform(kernels.make_mlp(kcfg["encoder_layers"], name="encoder"))
    decoder = hk.transform(kernels.make_mlp(kcfg["decoder_layers"], name="decoder"))

    optimizer = opts[cfg["optimizer_svgd"]]
    kwargs = {
        "target": targets[cfg["target"]](*cfg["target_args"]),
        "n_particles": cfg["n_particles"],
        "n_subsamples": cfg["n_subsamples"],
        "subsample_with_replacement": cfg["subsample_with_replacement"],
        "optimizer_svgd": svgd.Optimizer(
            *optimizer(*cfg["optimizer_svgd_args"])),
        "kernel": kernels.ard(logh=0),
        "encoder": encoder,
        "decoder": decoder,
        "lam": cfg["lam"],
    }
    return kwargs

def get_train_args(config):
    cfg = config["train_kernel"]
    optimizer = opts[cfg["optimizer_ksd"]]
    kwargs = {key: cfg[key] for key in ["n_iter", "ksd_steps", "svgd_steps"]}
    kwargs["key"] = random.PRNGKey(cfg["key"])
    kwargs["opt_ksd"] = svgd.Optimizer(*optimizer(*cfg["optimizer_ksd_args"]))
    return kwargs

def get_sample_args(config):
    kwargs = {
        "key": random.PRNGKey(config["train_kernel"]["key"]),
        "n_iter": config["train_kernel"]["n_iter"] * config["train_kernel"]["svgd_steps"]
    }
    return kwargs

def flat_to_nested(flat_dict):
    """takes in flat dict.
    returns dict with same nested structure as config."""
    out = dict()
    for k, v in config.items():
        out[k] = dict()
        for subk in v:
            if subk in flat_dict:
                out[k][subk] = flat_dict[subk]
            else:
                pass
#        if not out[k]:  # out[k] is empty
#           del out[k]
    # check everythings all right:
    for k in flat_dict:
        if not utils.nested_dict_contains_key(config, k):
            raise ValueError(f"Key {k} is not a configuration option.")
    return out
