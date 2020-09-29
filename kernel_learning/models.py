import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, tree_util, jacfwd, grad
from jax.experimental import optimizers
from jax.ops import index_update, index
import haiku as hk
import jax
import numpy as onp
import matplotlib.pyplot as plt
import optax

import traceback
import time
from tqdm import tqdm
from functools import partial
import json_tricks as json
import warnings

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
        self.time_waiting = 0
        self.min_validation_loss = None
        self.disable = patience == 0

    def update(self, validation_loss):
        """Returns True when early stopping criterion is met"""
        if self.min_validation_loss is None or self.min_validation_loss > validation_loss:
            self.min_validation_loss = validation_loss
            self.time_waiting = 0
        else:
            self.time_waiting += 1
        return

    def out_of_patience(self):
        return (self.time_waiting > self.patience) and not self.disable

    def reset(self):
        self.time_waiting = 0

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
                 n_particles: int = 120,
                 learning_rate=0.1,
                 optimizer="sgd",
                 num_groups=3,
                 noise_level=1.):
        """
        Arguments
        ----------
        gradient: takes in args (params, key, particles) and returns
            an array of shape (n, d), interpreted as grad(loss)(particles).
        proposal: instances of class distributions.Distribution
        """
        self.gradient = gradient
        self.target = target
        self.proposal = proposal
        self.n_particles = n_particles
        self.num_groups = num_groups

        # optimizer for particle updates
        self.opt = utils.optimizer_mapping[optimizer](learning_rate)
        self.threadkey, subkey = random.split(key)
        self.initialize_optimizer(subkey)
        self.step_counter = 0
        self.rundata = {}
#        rundata_keys = "step gradient_norm max_grad mean_grad median_grad mean std particles".split()
#        self.empty_stepdata = {k: None for k in rundata_keys}
        self.initialize_groups()

        # init noise scale
        self.noise_scales = np.ones(self.particles.shape)*learning_rate*noise_level
        self.noise_level = noise_level

    def initialize_groups(self, key=None):
        """Split particles into groups: training and validation"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        self.group_names = ("training", "validation", "test")[:self.num_groups] # TODO make this a dict or namedtuple
        key, subkey = random.split(key)
        idx = random.permutation(subkey, np.arange(self.n_particles))
        self.group_idx = idx.split(len(self.group_names))
        return None

    def reshuffle_tv(self, key=None):
        """Reshuffle indices of training and validation particles, keeping
        indices of test particles fixed."""
        assert len(self.group_idx) > 1
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
        self.particles = particles
        self.optimizer_state = self.opt.init(particles)
        return None

    def init_particles(self, key):
        particles = self.proposal.sample(self.n_particles, key=key)
        return particles

    def get_params(self, split_by_group=False):
        if split_by_group:
            return [self.particles[idx] for idx in self.group_idx]
        else:
            return self.particles

    @partial(jit, static_argnums=0)
    def perturb(self, key, particles, stepsizes):
        """Add Gaussian noise scaled proportionally to the step size, that is
        noise = sqrt(2 * stepsize) * eps, where eps is distributed as a standard
        normal.
        Returns:
            pertubed particles"""
        noise = random.normal(key, shape=particles.shape) * self.noise_level
        noise *= np.sqrt(2*np.abs(stepsizes) + 1e-8)
        return particles + noise

    def next_batch(self, key): # TODO make this a generator or something
        """Return next batch of particles (training and validation) for the
        training of a gradient field approximator. That is, take current
        particles, perturb, subsample etc (based on current optimizer state) and return."""
        assert len(self.group_idx) > 1
        particles = self.get_params()
        perturbed_particles = self.perturb(
            key, particles, stepsizes=self.noise_scales)
        return [perturbed_particles[idx] for idx in self.group_idx[:2]]

    @partial(jit, static_argnums=0)
    def _step(self, key, particles, optimizer_state, params, noise_scales):
        """
        Updates particles in direction of the gradient.
        Arguments:
            params: can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or None.
            noise_scales: shape (n, d), indicating scale of noise to be added to particles
        before the computation of gradient & the update.

        Returns:
            particles (updated)
            optimizer_state (updated)
            grad_aux: dict containing auxdata
        """
        key1, key2 = random.split(key)
        particles = self.perturb(key1, particles, noise_scales)
        grads, grad_aux = self.gradient(params, key2, particles, aux=True)
        grads = np.clip(grads, a_max=50, a_min=-50) # TODO: put in optimizer
        updated_grads, optimizer_state = self.opt.update(grads, optimizer_state, particles)
        effective_stepsizes = jax.tree_multimap(
            lambda a, b: a/(np.abs(b) + 1e-7), updated_grads, grads)
        particles = optax.apply_updates(particles, updated_grads)
        grad_aux.update({"grads": updated_grads})
        return particles, optimizer_state, grad_aux, effective_stepsizes

    def step(self, params, key=None):
        """Log rundata, take step. Mutates state"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        updated_particles, self.optimizer_state, auxdata, self.noise_scales = self._step(
            key, self.particles, self.optimizer_state, params, self.noise_scales)
        self.write_to_log(self._log(auxdata, self.particles, self.step_counter))
        self.particles = updated_particles
        self.step_counter += 1
        return None

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        """Append dictionary to log."""
        metrics.append_to_log(self.rundata, step_data)

    @partial(jit, static_argnums=0)
    def _log(self, auxdata, particles, step):
        gradient = auxdata["grads"]
        auxdata.update({
            "step": step,
            "gradient_norm": np.linalg.norm(gradient),
            "max_grad": np.max(np.abs(gradient)),
            "mean_grad": np.mean(np.abs(gradient)),
            "median_grad": np.median(np.abs(gradient)),
            "mean": np.mean(particles, axis=0),
            "std": np.std(particles, axis=0),
            "particles": particles,
        })
        for k, idx in zip(self.group_names, self.group_idx):
            # TODO: iterate thru particle groups directly instead
            auxdata.update({
                f"{k}_mean": np.mean(particles[idx], axis=0),
                f"{k}_std":  np.std(particles[idx], axis=0),
            })
        del auxdata["grads"]
        return auxdata

    def plot_mean_and_std(self, target=None, axs=None, **kwargs):
        """axs: two axes"""
        if target is None:
            target = self.target
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=[18,5])

        axs = axs.flatten()
        for ax, target_stat, stat_key in zip(axs, [target.mean, np.sqrt(np.diag(target.cov))], "mean std".split()):
            if target.d == 1:
                ax.axhline(y=target_stat, linestyle="--", color="green", label=f"Target {stat_key}")
            else:
                for y in target_stat:
                    ax.axhline(y=y, linestyle="--", color="green", label=f"Target {stat_key}")
                ax.plot(self.rundata[f"test_{stat_key}"], label=f"{stat_key}")
        return

    def plot_trajectories(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        p_over_time = np.array(self.rundata["particles"])
        ax.plot(p_over_time[:, :, 0], **kwargs)
        return ax

    def plot_final(self, target, ax=None, xlim=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        p = self.get_params()
        if target.d == 1:
            ax.hist(p[:, 0], density=True, alpha=0.5, label="Samples",   bins=25)
            plot.plot_fun(target.pdf, lims=ax.get_xlim(), ax=ax, label="Target density")
        elif target.d == 2:
            plot.scatter(p, ax=ax)
            plot.plot_fun_2d(target.pdf, xlims=ax.get_xlim(), ylims=ax.get_ylim(), ax=ax, **kwargs)
            plot.scatter(p, ax=ax)
        else:
            return NotImplementedError()
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend()
        return ax

    def animate_trajectory(self, target=None, fig=None, ax=None, interval=100, **kwargs):
        """Create animated scatterplot of particle trajectory"""
        trajectory = np.asarray(self.rundata["particles"])
        if target is not None:
            plot.plot_fun_2d(target.pdf, lims=(-20, 20), ax=ax)
        anim = plot.animate_array(trajectory, fig, ax, interval=interval)
        return anim


class GradientLearner():
    """
    Superclass. Learn a gradient vector field to transport particles.
    Need to implement the following methods:
    * loss_fn(self, params, particles) --> (loss, aux)
    * _log(self, particles, validation_particles, aux) --> step_log
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
        self.sizes = sizes if sizes is not None else [32, 32, target.d]
        if self.sizes[-1] != target.d:
            warnings.warn(f"Output dim must equal target dim; instead "
                          f"received output dim {sizes[-1]} and "
                          f"target dim {target.d}.")

        # init
        self.target = target
        self.lambda_reg = lambda_reg
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        self.mlp = nets.build_mlp(self.sizes, name="MLP",
                                  skip_connection=skip_connection,
                                  with_bias=True, activate_final=activate_final)
        self.opt = optax.adam(learning_rate)
        self.params = self.init_mlp()
        self.optimizer_state = self.opt.init(self.params)

        # other state
        self.learning_rate = learning_rate
        self.step_counter = 0
        self.rundata = {"train_steps": []}
        self.frozen_states = []
        self.patience = Patience(patience)

    def init_mlp(self, key=None, keep_params=False):
        """Initialize MLP"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.target.d)
        params = self.mlp.init(key, x_dummy)
        return params

    def get_params(self):
        return self.params

    def loss_fn(self, params, key, particles):
        raise NotImplementedError()

    def gradient(self, params, key, particles, aux=False):
        raise NotImplementedError()

    def _step_unjitted(self, key, params, optimizer_state, particles, validation_particles):
        """update parameters and compute validation loss"""
        [loss, loss_aux], grads = value_and_grad(self.loss_fn, has_aux=True)(params, key, particles)
        _, val_loss_aux = self.loss_fn(params, key, validation_particles)
        grads, optimizer_state = self.opt.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, grads)
        auxdata = (loss_aux, val_loss_aux, grads)
        return params, optimizer_state, auxdata
    _step = jit(_step_unjitted, static_argnums=0)

    def step(self, particles, validation_particles, disable_jit=False):
        """Step and mutate state"""
        self.threadkey, key = random.split(self.threadkey)
        step_fn = self._step_unjitted if disable_jit else self._step
        self.params, self.optimizer_state, auxdata = step_fn(key,
            self.params, self.optimizer_state, particles, validation_particles)
        self.write_to_log(
            self._log(particles, validation_particles, auxdata, self.step_counter))
        self.step_counter += 1
        return None

    def _log(self, particles, val_particles, auxdata, step_counter): # depends on loss_fn aux
        raise NotImplementedError()

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        metrics.append_to_log(self.rundata, step_data)

    def train(self, next_batch: callable, key=None, n_steps=100, progress_bar=False):
        """
        Arguments:
        * next_batch: callable, outputs next training batch. Signature:
            next_batch(key, noise_level=1.)
        """
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def step(key):
            key, subkey = random.split(key)
            train_x, val_x = next_batch(subkey)
            self.step(train_x, val_x)
            val_loss = self.rundata["validation_loss"][-1]
            self.patience.update(val_loss)
            return key

        for i in tqdm(range(n_steps), disable=not progress_bar):
            key = step(key)
            if self.patience.out_of_patience():
                self.patience.reset()
                break
        self.rundata["train_steps"].append(i+1)
        return

    def train_sampling_every_time(self, proposal, key=None, n_steps=100, batch_size=400,
                                  catch_nan_errors=False, progress_bar=True):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        key, subkey = random.split(key)
        validation_particles = proposal.sample(batch_size)
        for _ in tqdm(range(n_steps), disable=not progress_bar):
            try:
                key, subkey = random.split(key)
                particles = proposal.sample(batch_size, key=subkey)
                self.step(particles, validation_particles)
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

    def get_field(self, init_particles, params=None):
        """Return vector field v(\cdot) that approximates \nabla KL.
        This is used to compute particle updates x <-- x - eps * v(x)
        v = -f, where f maximizes the stein discrepancy.
        Arguments:
            init_particles: samples from current distribution, shape (n, d). These
            are used to compute mean and stddev for normalization."""
        if params is None:
            params = self.get_params()
        m = np.mean(init_particles)
        std = np.std(init_particles)
        def v(x):
            x_normalized = (x - m) / (std + 1e-4)
            return self.mlp.apply(params, None, x_normalized)
        return v

    def loss_fn(self, params, key, particles):
        f = utils.negative(self.get_field(particles, params))
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            particles, self.target.logpdf, f, aux=True)
        l2_f_sq = utils.l2_norm_squared(particles, f)
        loss = -stein_discrepancy + self.lambda_reg * l2_f_sq
        aux = [loss, stein_discrepancy, l2_f_sq, stein_aux]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        train_aux, val_aux, g = aux
        loss, sd, l2v, stein_aux = train_aux
        drift, repulsion = stein_aux # shape (2, d)
        val_loss, val_sd, *_ = val_aux
        gradient_norms = [np.linalg.norm(v) for v in jax.tree_leaves(g)]
        step_log = {
            "step_counter": step_counter,
            "training_loss": loss,
            "training_sd": sd,
            "validation_loss": val_loss,
            "validation_sd": val_sd,
            "l2_norm": l2v,
            "mean_drift": np.mean(drift),
            "mean_repulsion": np.mean(repulsion),
            "gradient_norms": gradient_norms,
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
    def __init__(self, target, num_leaders=0, **kwargs):
        super().__init__(target, **kwargs)
        self.activation_kernel = kernels.get_rbf_kernel(1)
        self.num_leaders = num_leaders if num_leaders > 0 else 10**7
        self.params = (self.init_mlp(), 1.)
        self.optimizer_state = self.opt.init(self.params)

    def get_field(self, inducing_particles, params=None):
        """return -phistar(\cdot)"""
        if params is None:
            params = self.get_params()
        kernel = self.get_kernel(inducing_particles, params)
        def phistar(x):
            return stein.phistar_i(x, inducing_particles, self.target.logpdf, kernel, aux=False)
        return utils.negative(phistar)

    def loss_fn(self, params, key, particles):
        # *params, leaders = params # keep inducing particles fixed
        particles = random.permutation(key, particles)
        leaders = particles[:self.num_leaders]
        phi = utils.negative(self.get_field(leaders, params=params))
        kernel = self.get_kernel(leaders)
        ksd_squared = stein.stein_discrepancy(particles, self.target.logpdf, phi)
        #ksd_squared = stein.ksd_squared_l(particles, self.target.logpdf, kernel)
        #ksd_squared = stein.ksd_squared_u(particles, self.target.logpdf, kernel)
        l2_squared = utils.l2_norm_squared(particles, phi)
        loss = -ksd_squared + self.lambda_reg * l2_squared
        aux = [loss, ksd_squared, l2_squared]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        names = "loss ksd l2_norm".split()
        step_log = {"step_counter": step_counter}
        for group, data in zip(["training", "validation"], aux[:-1]):
            step_log.update({
                group+"_"+name: value for name, value in zip(names, data)
            })
        return step_log

    def gradient(self, params, _, particles, aux=False):
        v = self.get_field(inducing_particles=particles, params=params)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)

    def get_kernel(self, init_particles, params=None):
        """
        Arguments:
        init_particles: samples from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        if params is None:
            params = self.get_params()
        mlp_params, scale = params
        m = np.mean(init_particles)
        std = np.std(init_particles)
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
    def __init__(self, target, lam=0., **kwargs):
        super().__init__(target, **kwargs)
        self.lam=lam

    def get_score(self, init_particles, params=None):
        """Return callables approximating score of q = grad(log q)
        Arguments:
        init_particles: particles from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        if params is None:
            params = self.get_params()
        m = np.mean(init_particles)
        std = np.std(init_particles)
        def score(x):
            x_normalized = (x - m) / (std + 1e-4)
            return self.mlp.apply(params, None, x_normalized)
        return score

    def get_field(self, init_particles, params=None):
        """Return vector field v(\cdot) that approximates
        \nabla KL = grad(log q) - grad(log p). This is used to compute
        particle updates x <-- x - eps * v(x)

        Arguments:
        init_particles: samples from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        score = self.get_score(init_particles=init_particles, params=params)

        def v(x):
            return score(x) - grad(self.target.logpdf)(x) / (2*self.lambda_reg)
        return v

    def loss_fn(self, params, key, particles):
        """Evaluates E[div(score)] + lambda * l2_norm(score)^2"""
        score = self.get_score(particles, params=params)
        mean_divergence  = np.mean(vmap(utils.div(score))(particles))
        mean_squared_div = np.mean(vmap(utils.div_sq(score))(particles))
        l2_norm_sq = utils.l2_norm_squared(particles, score)
        loss = mean_divergence + self.lambda_reg * l2_norm_sq + self.lam * mean_squared_div
        aux = [loss, mean_divergence, l2_norm_sq]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        train_aux, val_aux, g = aux
        loss, mean_div, l2 = train_aux
        val_loss, val_mean_div, val_l2 = val_aux
        layer_norms = [np.linalg.norm(v) for v in tree_util.tree_leaves(g)]
        step_log = {
            "step_counter": step_counter,
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
        self.rundata = {}

    def get_field(self, inducing_particles, aux=False):
        """return -phistar(\cdot)"""
        def phistar(x):
            return stein.phistar_i(x, inducing_particles, self.target.logpdf, self.kernel)
        return utils.negative(phistar)

    def gradient(self, params, key, particles, aux=False, scaled=False):
        """Compute approximate KL gradient.
        params and key args are not used."""
        v = self.get_field_scaled(particles) if scaled else self.get_field(particles)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)

    def get_field_scaled(self, inducing_particles):
        phi = stein.get_phistar(self.kernel, self.target.logpdf, inducing_particles)
        l2_phi_squared = utils.l2_norm_squared(inducing_particles, phi)
        ksd = stein.stein_discrepancy(inducing_particles, self.target.logpdf, phi)
        alpha = ksd / (2*self.lambda_reg*l2_phi_squared)
        return utils.mul(phi, -alpha)


class KernelizedScoreMatcher():
    """Compute kernelized score estimate"""
    def __init__(self,
                target,
                key=random.PRNGKey(42),
                kernel = kernels.get_rbf_kernel(1),
                 lambda_reg=1/2,
                 scale=1.):
        self.target = target
        self.threadkey, subkey = random.split(key)
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.scale=scale
        self.rundata = {}

    def objective(self, particles, f):
        divergence_term = -np.mean(vmap(utils.div(f))(particles))
        return divergence_term

    def target_score(self, x):
        return grad(self.target.logpdf)(x) / (2*self.lambda_reg)

    def get_score(self, inducing_particles):
        def kernelized_score(x):
            """Return E_y[grad(kernel)(y, x)]"""
            return -np.mean(vmap(grad(self.kernel), (0, None))(inducing_particles, x), axis=0)

        M = self.objective(inducing_particles, kernelized_score)
        scale = M / (utils.l2_norm_squared(inducing_particles, kernelized_score) * 2 * self.lambda_reg)
        kernelized_score = utils.mul(kernelized_score, scale * self.scale)
        return kernelized_score

    def get_kernelized_target_score(self, inducing_particles):
        """Not using this yet"""
        def kernelized_score(x):
            def inner(x_, y):
                return grad(target.logpdf)(y) * self.kernel(y, x_)
            return -np.mean(vmap(inner)(x, inducing_particles), axis=0)
        M = self.objective(inducing_particles, kernelized_score)
        scaler = M / (utils.l2_norm_squared(inducing_particles, kernelized_score) * 2 * self.lambda_reg)
        kernelized_score = utils.mul(kernelized_score, scaler)
        return kernelized_score

    def get_field(self, inducing_particles):
        """Return vector field v(\cdot) that approximates
        \nabla KL = grad(log q) - grad(log p).
        This is used to compute particle updates x <-- x - eps * v(x)"""
        score = self.get_score(inducing_particles)

        def v(x):
            return score(x) - self.target_score(x)
        return v

    def gradient(self, params, key, particles, aux=False):
        """Compute approximate KL gradient"""
        score = self.get_score(particles)
        score_batched = vmap(score)(particles)
        energy_batched = vmap(self.target_score)(particles)
        aux = {"score_norm": np.linalg.norm(score_batched),
               "target_score_norm": np.linalg.norm(energy_batched)}
        if aux:
            return score_batched - energy_batched, aux
        else:
            return score_batched - energy_batched


class EnergyGradient():
    """Compute pure SGLD gradient $\nabla \log p(x)$ (without noise)"""
    def __init__(self,
                target,
                key=random.PRNGKey(42),
                lambda_reg=1/2):
        self.target = target
        self.threadkey, subkey = random.split(key)
        self.lambda_reg = lambda_reg
        self.rundata = {}

    def target_score(self, x):
        return grad(self.target.logpdf)(x) / (2*self.lambda_reg)

    def get_field(self, inducing_particles):
        """Return vector field used for updating, $\nabla \log p(x)$
        (without noise)."""
        return utils.negative(self.target_score)

    def gradient(self, params, key, particles, aux=False):
        """Compute gradient used for SGD particle update"""
        v = self.get_field(particles)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)
