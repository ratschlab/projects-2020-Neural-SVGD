import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, tree_util, jacfwd, grad
from jax.experimental import optimizers
from jax.ops import index_update, index
import haiku as hk
import jax
import numpy as onp
import matplotlib.pyplot as plt

import traceback
import time
from tqdm import tqdm
from functools import partial
import json_tricks as json

import utils
import metrics
import stein
import kernels
import nets
import distributions
import plot

from typing import Mapping
import os
on_cluster = not os.getenv("HOME") == "/home/lauro"
disable_tqdm = on_cluster
eps = 1e-4


class Logger():
    def __init__(self):
        self.data = {}

    def write(data: Mapping[str, np.ndarray], reducer: callable=None):
        if reducer is not None:
            data = {k: reducer(v) for k, v in data.items()}
        metrics.append_to_log(self.data, data)
        return

    def reset(self):
        self.data = {}


class Optimizer():
    def __init__(self, opt_init, opt_update, get_params):
        """opt_init, opt_update, get_params are the three functions obtained
        from a stax.optimizer call."""
        self.init = jit(opt_init)
        self.update = jit(opt_update)
        self.get_params = jit(get_params)


class Patience():
    """Criterion for early stopping"""
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.impatience = 0
        self.min_validation_loss = None
        self.disable = patience == 0

    def update(self, validation_loss):
        """Returns True when early stopping criterion is met"""
        if self.min_validation_loss is None or self.min_validation_loss > validation_loss:
            self.min_validation_loss = validation_loss
            self.impatience = 0
        else:
            self.impatience += 1
        return

    def out_of_patience(self):
        return (self.impatience > self.patience) and not self.disable

    def reset(self):
        self.impatience = 0

class Particles():
    """
    Container class for particles, particle optimizer,
    particle update step method, and particle metrics.
    """
    def __init__(self,
                 key,
                 gradient: callable,
                 proposal,
                 target=None,
                 n_particles: int = 400,
                 learning_rate=0.1):
        """
        Arguments
        ----------
        gradient: takes in args (params, key, particles) and returns
            an array of shape (n, d), interpreted as grad(loss)(particles).
        proposal: instances of class distributions.Distribution
        optimizer needs to have pure methods
           optimizer.init(params) -> state
           optimizer.update(key, gradient, state) -> state
           optimizer.get_params(state) -> params
        """
        self.gradient = gradient
        self.target = target
        self.proposal = proposal
        self.n_particles = n_particles

        # optimizer for particle updates
        self.opt = Optimizer(*optimizers.sgd(learning_rate))
        self.threadkey, subkey = random.split(key)
        self.initialize_optimizer(subkey)
        self.step_counter = 0
        self.rundata = {}
        self.initialize_groups()

    def initialize_groups(self, key=None):
        """Split particles into groups: training and validation"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        self.group_names = ("training", "validation", "test") # TODO make this a dict or namedtuple; add 'test'
        key, subkey = random.split(key)
        idx = random.permutation(subkey, np.arange(self.n_particles))
        self.group_idx = idx.split(len(self.group_names))
        return None

    def reshuffle_tv(self, key=None):
        """Reshuffle indices of training and validation particles"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        t, v, _ = self.group_idx
        tv = np.concatenate([t, v])
        t, v = random.permutation(key, tv).split(2)
        self.group_idx[0] = t
        self.group_idx[1] = v
        return

    def initialize_optimizer(self, key=None, particles=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        if particles is None:
            particles = self.init_particles(key)
        self.optimizer_state = self.opt.init(particles)
        return None

    def init_particles(self, key):
        particles = self.proposal.sample(self.n_particles, key=key)
        return particles

    def get_params(self, split_by_group=False):
        if split_by_group:
            return [self.opt.get_params(self.optimizer_state)[idx]
                    for idx in self.group_idx]
        else:
            return self.opt.get_params(self.optimizer_state)

    def perturb(self, noise_level=1e-2, key=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        particles = self.get_params()
        particles += random.normal(key, shape=particles.shape)
        self.initialize_optimizer(particles=particles)
        return None

    @partial(jit, static_argnums=0)
    def _step(self, key, optimizer_state, params, step_counter, noise_pre=0):
        """
        Updates particles in direction of the gradient.

        params can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or nothing.
        """
        particles = self.opt.get_params(optimizer_state)
        particles = particles + noise_pre * random.normal(key, particles.shape)
        gradient, grad_aux = self.gradient(params, key, particles, aux=True)
        gradient = np.clip(gradient, a_max=50, a_min=-50)
        optimizer_state = self.opt.update(step_counter, gradient, optimizer_state)
        auxdata = gradient
        return optimizer_state, auxdata, grad_aux # grad_aux is a dict

    def step(self, params, key=None, noise_pre=0):
        """Log rundata, take step, update loglikelihood. Mutates state"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        updated_optimizer_state, auxdata, grad_aux = self._step(
            key, self.optimizer_state, params, self.step_counter, noise_pre)
        self.log(auxdata, grad_aux)
        self.optimizer_state = updated_optimizer_state # take step
        self.step_counter += 1
        return None

    def log(self, auxdata, grad_aux):
        gradient = auxdata
        particles = self.get_params()
        metrics.append_to_log(self.rundata, grad_aux)
        metrics.append_to_log(self.rundata, {
            "step": self.step_counter,
            "gradient_norm": np.linalg.norm(gradient),
            "max_grad": np.max(np.abs(gradient)),
            "mean_grad": np.mean(np.abs(gradient)),
            "median_grad": np.median(np.abs(gradient)),
            "mean": np.mean(particles, axis=0),
            "std": np.std(particles, axis=0),
            "particles": particles,
        })
        for k, idx in zip(self.group_names, self.group_idx): # TODO: iterate thru particle groups directly instead
            metrics.append_to_log(self.rundata, {
                f"{k}_mean": np.mean(particles[idx], axis=0),
                f"{k}_std":  np.std(particles[idx], axis=0),
            })
        return

    def plot_mean_and_std(self, target=None, axs=None, **kwargs):
        """axs: two axes"""
        if target is None:
            target = self.target
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=[18,5])

        particles = self.get_params()
        axs = axs.flatten()
        for ax, target_stat, stat_key in zip(axs, [target.mean, np.sqrt(np.diag(target.cov))], "mean std".split()):
            ax.axhline(y=target_stat, linestyle="--", color="green", label=f"Target {stat_key}")
            ax.plot(self.rundata[f"test_{stat_key}"], label=f"{stat_key}")
#            for group_name in self.group_names:
#                ax.plot(self.rundata[f"{group_name}_{stat_key}"], label=f"{stat_key}")
#                ax.legend()
        return

    def plot_trajectories(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        p_over_time = np.array(self.rundata["particles"])
        ax.plot(p_over_time[:, :, 0], **kwargs)
        return ax

    def plot_final(self, ax=None, target=None, xlim=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        p = self.get_params()
        ax.hist(p[:, 0], density=True, alpha=0.5, label="Samples",   bins=25)
        if target is not None:
            plot.plot_fun(target.pdf, lims=ax.get_xlim(), ax=ax, label="Target density")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend()
        return ax

class GradientLearner():
    """
    Superclass. Learn a gradient vector field to transport particles.
    Need to implement the following methods:
    * loss_fn(self, params, samples) --> (loss, aux)
    * _log(self, samples, validation_samples, aux) --> step_log
    * gradient(self, params, key, particles, aux=False) --> particle_gradient
    """
    def __init__(self,
                 target,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 skip_connection=False,
                 activate_final=False,
                 learning_rate: float = 1e-2,
                 lambda_reg: float = 1/2,
                 patience: int = 20):
        # warnings
        if sizes[-1] != target.d:
            raise ValueError(f"Output dim must equal target dim; instead "
                             f"received output dim {sizes[-1]} and "
                             f"target dim {target.d}.")

        # init
        self.sizes = sizes if sizes is not None else [32, 32, target.d]
        self.target = target
        self.lambda_reg = lambda_reg
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        self.mlp = nets.build_mlp(self.sizes, name="MLP",
                                  skip_connection=skip_connection,
                                  with_bias=True, activate_final=activate_final)
        self.opt = Optimizer(*optimizers.adam(learning_rate))
        self.step_counter = 0
        self.initialize_optimizer(subkey)
        self.rundata = {"train_steps": []}
        self.frozen_states = []
        self.patience = Patience(patience)


    def initialize_optimizer(self, key=None, keep_params=False):
        """Initialize optimizer. If keep_params=True, then only the learning
        rate schedule is reinitialized, otherwise the model parameters are
        also reinitialized."""
        if keep_params:
            self.optimizer_state = self.opt.init(self.get_params())
        else:
            x_dummy = np.ones(self.target.d)
            if key is None:
                self.threadkey, subkey = random.split(self.threadkey)
            else:
                key, subkey = random.split(key)
            init_params = self.mlp.init(subkey, x_dummy)
            self.optimizer_state = self.opt.init(init_params)
        return None

    def get_params(self):
        return self.opt.get_params(self.optimizer_state)

    def loss_fn(self, params, samples):
        raise NotImplementedError()

    def gradient(self, params, key, particles, aux=False):
        raise NotImplementedError()

    def _step_unjitted(self, optimizer_state, samples, validation_samples, step: int):
        """update parameters and compute validation loss"""
        params = self.opt.get_params(optimizer_state)
        [loss, loss_aux], g = value_and_grad(self.loss_fn, has_aux=True)(params, samples)
        _, val_loss_aux = self.loss_fn(params, validation_samples)
        optimizer_state = self.opt.update(step, g, optimizer_state)
        auxdata = (loss_aux, val_loss_aux, g)
        return optimizer_state, auxdata
    _step = jit(_step_unjitted, static_argnums=0)

    def step(self, samples, validation_samples, disable_jit=False):
        """Step and mutate state"""
        step_fn = self._step_unjitted if disable_jit else self._step
        updated_optimizer_state, auxdata = step_fn(
            self.optimizer_state, samples, validation_samples, self.step_counter)
        self.optimizer_state = updated_optimizer_state
        self.write_to_log(self._log(samples, validation_samples, auxdata))
        self.step_counter += 1
        return None

    def _log(self, aux): # depends on loss_fn aux
        raise NotImplementedError()

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        metrics.append_to_log(self.rundata, step_data)

    def train(self, samples, validation_samples, key=None, n_steps=100, noise_level=0, progress_bar=False):
        """
        Arguments:
        * samples: batch to train on
        * validation_samples
        """
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def step(key, samples):
            step_samples = samples + random.normal(key, samples.shape)*noise_level
            step_samples_val = validation_samples + random.normal(
                key, validation_samples.shape)*noise_level
            self.step(step_samples, step_samples_val)

        for i in tqdm(range(n_steps), disable=not progress_bar):
            key, subkey = random.split(key)
            step(subkey, samples)
            val_loss = self.rundata["validation_loss"][-1]
            self.patience.update(val_loss)
            if self.patience.out_of_patience():
                self.patience.reset()
                break
        self.rundata["train_steps"].append(i)
        return

    def train_sampling_every_time(self, proposal, key=None, n_steps=100, batch_size=400,
                                  catch_nan_errors=False, progress_bar=True):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        key, subkey = random.split(key)
        validation_samples = proposal.sample(batch_size)
        for _ in tqdm(range(n_steps), disable=not progress_bar):
            try:
                key, subkey = random.split(key)
                samples = proposal.sample(batch_size, key=subkey)
                self.step(samples, validation_samples)
            except FloatingPointError as err:
                if catch_nan_errors:
                    return
                else:
                    raise err from None
        return

    def freeze_state(self):
        """Stores current state as tuple (step_counter, params, rundata)"""
        self.frozen_states.append((self.step_counter,
                                   self.get_params(),
                                   self.rundata))
        return


class SDLearner(GradientLearner):
    """Parametrize function to maximize the stein discrepancy"""
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)

    def get_field(self, init_samples, params=None):
        """Return vector field v(\cdot) that approximates \nabla KL.
        This is used to compute particle updates x <-- x - eps * v(x)
        v = -f, where f maximizes the stein discrepancy.
        Arguments:
            init_samples: samples from current distribution, shape (n, d). These
            are used to compute mean and stddev for normalization."""
        if params is None:
            params = self.get_params()
        m = np.mean(init_samples)
        std = np.std(init_samples)
        def v(x):
            x_normalized = (x - m) / (std + 1e-4)
            return self.mlp.apply(params, None, x_normalized)
        return v

    def loss_fn(self, params, samples):
        f = utils.negative(self.get_field(samples, params))
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            samples, self.target.logpdf, f, aux=True)
        l2_f_sq = utils.l2_norm_squared(samples, f)
        loss = -stein_discrepancy + self.lambda_reg * l2_f_sq
        aux = [loss, stein_discrepancy, l2_f_sq, stein_aux]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, samples, validation_samples, aux):
        train_aux, val_aux, g = aux
        loss, sd, l2v, stein_aux = train_aux
        drift, repulsion = stein_aux # shape (2, d)
        val_loss, val_sd, *_ = val_aux
        layer_norms = [np.linalg.norm(v) for v in tree_util.tree_leaves(g)]
        step_log = {
            "step": self.step_counter,
            "training_loss": loss,
            "training_sd": sd,
            "validation_loss": val_loss,
            "validation_sd": val_sd,
            "l2_norm": l2v,
            "mean_drift": np.mean(drift),
            "mean_repulsion": np.mean(repulsion),
            "gradient_norms": layer_norms,
        }
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(particles, params)
        if aux:
            stein_discrepancy, stein_aux = stein.stein_discrepancy(
                particles, self.target.logpdf, v, aux=True)
            auxdata = {"sd": stein_discrepancy}
            return v(particles), auxdata
        else:
            return v(particles)


class KernelLearner(GradientLearner):
    """Parametrize kernel to learn the KSD"""
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.activation_kernel = kernels.get_rbf_kernel(1)

    def initialize_optimizer(self, key=None, keep_params=False):
        """Initialize optimizer. If keep_params=True, then only the learning
        rate schedule is reinitialized, otherwise the model parameters are
        also reinitialized."""
        if keep_params:
            self.optimizer_state = self.opt.init(self.get_params())
        else:
            x_dummy = np.ones(self.target.d)
            if key is None:
                self.threadkey, subkey = random.split(self.threadkey)
            else:
                key, subkey = random.split(key)
            init_params_enc = self.mlp.init(subkey, x_dummy)
            init_params = (init_params_enc, 1.) # (mlp params, scaling_param)
            self.optimizer_state = self.opt.init(init_params)
        return None

    def get_field(self, inducing_samples, params=None):
        """return -phistar(\cdot)"""
        if params is None:
            params = self.get_params()
        kernel = self.get_kernel(inducing_samples, params)
        def phistar(x):
            return stein.phistar_i(x, inducing_samples, self.target.logpdf, kernel, aux=False)
        return utils.negative(phistar)

    def loss_fn(self, params, samples):
        kernel = self.get_kernel(samples, params)
        ksd_squared = stein.ksd_squared_u(
            samples, self.target.logpdf, kernel, include_stddev=False)
        kxx = np.mean(vmap(kernel)(samples, samples))
        phi = self.get_field(samples, params=params)
        loss = -ksd_squared + self.lambda_reg * utils.l2_norm_squared(samples, phi)
        aux = [loss, ksd_squared, kxx]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, samples, validation_samples, aux):
        names = "loss ksd kxx".split()
        step_log = {}
        for group, data in zip(["training", "validation"], aux[:-1]):
            step_log.update({
                group+"_"+name: value for name, value in zip(names, data)
            })
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(params, inducing_samples=particles)
        if aux:
            return vmap(v)(particles), None
        else:
            return vmap(v)(particles)

    def get_kernel(self, init_samples, params=None):
        """
        Arguments:
        init_samples: samples from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        if params is None:
            params = self.get_params()
        mlp_params, scale = params
        m = np.mean(init_samples)
        std = np.std(init_samples)
        def kernel(x, y):
            x, y = np.asarray(x), np.asarray(y)
            x_normalized = (x - m) / (std + 1e-4)
            y_normalized = (y - m) / (std + 1e-4)
            k = self.activation_kernel(self.mlp.apply(mlp_params, None, x_normalized),
                                       self.mlp.apply(mlp_params, None, y_normalized))
            return scale * k
        return kernel


class ScoreLearner(GradientLearner):
    """Neural score matching"""
    def __init__(self, target, lam=0.1, **kwargs):
        super().__init__(target, **kwargs)
        self.lam=lam

    def get_score(self, init_samples, params=None):
        """Return callable s approximating score of q = grad(log q)
        Arguments:
        init_samples: samples from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        if params is None:
            params = self.get_params()
        m = np.mean(init_samples)
        std = np.std(init_samples)
        def score(x):
            x_normalized = (x - m) / (std + 1e-4)
            return self.mlp.apply(params, None, x_normalized)
        return score

    def get_field(self, init_samples, params=None):
        """Return vector field v(\cdot) that approximates
        \nabla KL = grad(log q) - grad(log p). This is used to compute
        particle updates x <-- x - eps * v(x)

        Arguments:
        init_samples: samples from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        score = self.get_score(init_samples=init_samples, params=params)

        def v(x):
            return score(x) - grad(self.target.logpdf)(x) / (2*self.lambda_reg)
        return v

    def loss_fn(self, params, samples):
        """Evaluates E[div(score)] + lambda * l2_norm(score)^2"""
        score = self.get_score(samples, params=params)
        mean_divergence  = np.mean(vmap(utils.div(score))(samples))
        mean_squared_div = np.mean(vmap(utils.div_sq(score))(samples))
        l2_norm_sq = utils.l2_norm_squared(samples, score)
        loss = mean_divergence + self.lambda_reg * l2_norm_sq + self.lam * mean_squared_div
        aux = [loss, mean_divergence, l2_norm_sq]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, samples, validation_samples, aux):
        train_aux, val_aux, g = aux
        loss, mean_div, l2 = train_aux
        val_loss, val_mean_div, val_l2 = val_aux
        layer_norms = [np.linalg.norm(v) for v in tree_util.tree_leaves(g)]
        step_log = {
            "step": self.step_counter,
            "training_loss": loss,
            "training_div": mean_div,
            "validation_div": val_mean_div,
            "validation_loss": val_loss,
            "training_l2_norm": l2,
            "validation_l2_norm": val_l2,
            "gradient_norms": layer_norms,
        }
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(particles, params=params)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)


class KernelGradient():
    """Computes the SVGD approximation to grad(KL), ie
    phi*(y) = E[grad(log p)(y) k(x, y) + div(k)(x, y)]"""
    def __init__(self,
                target,
                key=random.PRNGKey(42),
                kernel = kernels.get_rbf_kernel(1),
                lambda_reg = 1/2):
        self.target = target
        self.threadkey, subkey = random.split(key)
        self.kernel = kernel
        self.lambda_reg = lambda_reg

    def get_field(self, inducing_samples):
        """return -phistar(\cdot)"""
        def phistar(x):
            return stein.phistar_i(x, inducing_samples, self.target.logpdf, self.kernel, aux=False)
        return utils.negative(phistar)

    def gradient(self, params, key, particles, aux=False, scaled=False):
        """Compute approximate KL gradient.
        params and key args are not used."""
        v = self.get_field_scaled(particles) if scaled else self.get_field(particles)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)

    def get_field_scaled(self, inducing_samples):
        phi = stein.get_phistar(self.kernel, target.logpdf, inducing_samples)
        l2_phi_squared = utils.l2_norm_squared(inducing_samples, phi)
        ksd = stein.stein_discrepancy(inducing_samples, target.logpdf, phi)
        alpha = ksd / (2*self.lambda_reg*l2_phi_squared)
        return utils.mul(phi, -alpha)


class KernelizedScoreMatcher():
    """Compute kernelized score estimate"""
    def __init__(self,
                target,
                key=random.PRNGKey(42),
                kernel = kernels.get_rbf_kernel(1),
                lambda_reg=1/2):
        """Score estimate is scaled such that if the true score is in
        the RKHS, then l2_norm(score_estimate) = l2_norm(grad(log q) / (2*lambda_reg))"""
        self.target = target
        self.threadkey, subkey = random.split(key)
        self.kernel = kernel
        self.lambda_reg = lambda_reg

    def objective(self, samples, f):
        divergence_term = -np.mean(vmap(utils.div(f))(samples))
        return divergence_term

    def target_score(self, x):
        return grad(self.target.logpdf)(x) / (2*self.lambda_reg)

    def get_score(self, inducing_samples):
        def kernelized_score(x):
            return -np.mean(vmap(grad(self.kernel), (0, None))(inducing_samples, x))

        # rescale s.t. l2_norm(kernelized_score) = M / 2*lambda
        kernelized_score = utils.l2_normalize(kernelized_score, inducing_samples)
        M = self.objective(inducing_samples, kernelized_score)
        self.scaler = M / (2*self.lambda_reg)
#         M = np.clip(M, 0)
        kernelized_score = utils.l2_normalize(
            kernelized_score, inducing_samples, M / (2*self.lambda_reg))
        return kernelized_score

    def get_field(self, inducing_samples):
        """Return vector field v(\cdot) that approximates
        \nabla KL = grad(log q) - grad(log p).
        This is used to compute particle updates x <-- x - eps * v(x)"""
        score = self.get_score(inducing_samples)

        def v(x):
            return score(x) - self.target_score(x)
        return v

    def gradient(self, params, key, particles, aux=False):
        """Compute approximate KL gradient"""
        v = self.get_field(particles)
        if aux:
            return v(particles), {}
        else:
            return v(particles)
