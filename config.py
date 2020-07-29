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

    "encoder_layers": [4, 4, 2],
    "decoder_layers": [4, 4, 2],
    "kernel": "ard", # no alternatives atm
}

config["train_kernel"] = {
    "key": 0,
    "n_iter": 5,
    "ksd_steps": 1,
    "svgd_steps": 1,
    "optimizer_ksd": "Adam",  # One of ["Adam", "Adagrad", "SGD"]. optimizer for encoder and decoder
    "optimizer_ksd_args": [0.03],
    "lambda_reg": 1.0,
    "train": True # if false, set some args to null and
                  # just do a vanilla run w/o kernel learning.
}

####### utilities
opts = {
    "Adam": optimizers.adam,
    "Adagrad": optimizers.adagrad,
    "SGD": optimizers.sgd,
    "RMSProp": optimizers.rmsprop,
}

def get_svgd_args(config):
    """config is entire config (tho also works if you only pass the svgd subdict)"""
    if "svgd" in config:
        svgd_config = svgd_config["svgd"]
        train = config["train_kernel"]["train"]
    else:
        svgd_config = config
        train = True # assume
    targets = {
        "Gaussian": metrics.Gaussian,
        "Gaussian Mixture": metrics.GaussianMixture
    }

    if train:
        encoder = hk.transform(kernels.make_mlp(svgd_config["encoder_layers"], name="encoder"))
        decoder = hk.transform(kernels.make_mlp(svgd_config["decoder_layers"], name="decoder"))
    else:
        encoder = hk.transform(lambda x: x)
        decoder = None

    optimizer = opts[svgd_config["optimizer_svgd"]]
    kwargs = {
        "target": targets[svgd_config["target"]](*svgd_config["target_args"]),
        "n_particles": svgd_config["n_particles"],
        "n_subsamples": svgd_config["n_subsamples"],
        "subsample_with_replacement": svgd_config["subsample_with_replacement"],
        "optimizer_svgd": svgd.Optimizer(
            *optimizer(*svgd_config["optimizer_svgd_args"])),
        "kernel": kernels.ard(logh=0),
        "encoder": encoder,
        "decoder": decoder,
    }
    if kwargs["target"].d != svgd_config["decoder_layers"][-1] and train:
        warnings.warn(f"The size of the last layer of the decoder must equal"
        "the target particle dimension d={kwargs['target'].d}."
        "Instead received layer size {svgd_config[\"decoder_layers\"][-1]}. I'm"
        "modifying the last decoder layer so that it fits.")
        svgd_config["decoder_layers"][-1] = kwargs["target"].d
        kwargs["decoder"] = hk.transform(kernels.make_mlp(svgd_config["decoder_layers"], name="decoder"))
    return kwargs

def get_train_args(train_config):
    """train_coinfig is the subdict config["train_kernel"]"""
    if "train_kernel" in train_config:
        train_config = train_config["train_kernel"]
    if not train_config["train"]:
        raise ValueError(f"Configuration option 'train' must be set to True if you"
        "want to train the encoder, but it's currently set to False.")
    optimizer = opts[train_config["optimizer_ksd"]]
    kwargs = {key: train_config[key] for key in ["n_iter", "ksd_steps", "svgd_steps"]}
    kwargs["key"] = random.PRNGKey(train_config["key"])
    kwargs["opt_ksd"] = svgd.Optimizer(*optimizer(*train_config["optimizer_ksd_args"]))
    kwargs["lambda_reg"] = train_config["lambda_reg"]
    return kwargs

def get_sample_args(train_config):
    """train_config is the subdict train_config["train_kernel"].
    Need to pass encoder_params separately if train=True."""
    if "train_kernel" in train_config:
        train_config = train_config["train_kernel"]
    kwargs = {
        "key": random.PRNGKey(train_config["key"]),
        "n_iter": train_config["n_iter"] * train_config["svgd_steps"]
    }
    if not train_config["train"]:
        kwargs["encoder_params"] = {}
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
