import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, tree_util, jacfwd, grad
from jax.experimental import optimizers
import haiku as hk
import jax
import optax
import chex
from dataclasses import astuple, asdict

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


class Logger:
    def __init__(self):
        self.data = {}

    def write(data: Mapping[str, np.ndarray], reducer: callable=None):
        if reducer is not None:
            data = {k: reducer(v) for k, v in data.items()}
        metrics.append_to_log(self.data, data)
        return

    def reset(self):
        self.data = {}


class Patience:
    """Criterion for early stopping"""
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.time_waiting = 0
        self.min_validation_loss = None
        self.disable = patience == -1

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

    def reset(self, patience=None):
        self.time_waiting = 0
        self.min_validation_loss = None
        if patience:
            self.patience = patience


@chex.dataclass
class SplitData:
    """Training-test split"""
    training: np.ndarray
    test: np.ndarray = None

    def __iter__(self):
        return iter([p for p in astuple(self) if p is not None])

    def items(self):
        return ((k, v) for k, v in asdict(self).items() if v is not None)

    def keys(self):
        return (k for k, _ in self.items())

    def __add__(self, data):
        assert all([k1 == k2 for k1, k2 in zip(self.keys(), data.keys())])
        return SplitData(*[a + b for a, b in zip(self, data)])

class Particles:
    """
    Container class for particles, particle optimizer,
    particle update step method, and particle metrics.
    """
    def __init__(self,
                 key,
                 gradient: callable,
                 init_samples,
                 target=None,
                 n_particles: int = 50,
                 learning_rate=1e-2,
                 optimizer="sgd",
                 num_groups=2,
                 noise_level=0.):
        """
        Arguments
        ----------
        gradient: takes in args (params, key, particles) and returns
            an array of shape (n, d), interpreted as grad(loss)(particles).
        init_samples: either a callable sample(num_samples, key), or an nd.array
        of shape (n, d) containing initial samples.
        """
        self.gradient = gradient
        self.target = target
        self.n_particles = n_particles
        self.num_groups = num_groups
        self.threadkey, subkey = random.split(key)
        self.init_samples = init_samples
        self.particles = self.init_particles(subkey)

        # optimizer for particle updates
        self.optimizer_str = optimizer
        self.learning_rate = learning_rate
        self.opt = utils.optimizer_mapping[optimizer](learning_rate)
        self.optimizer_state = self.opt.init(self.particles)
        self.step_counter = 0
        self.rundata = {}
        self.noise_level = noise_level

    def init_particles(self, key):
        """Returns namedtuple with training and test particles"""
        assert self.num_groups <= 2 # SplitData only supports two groups
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        keys = random.split(key, self.num_groups)
        try:
            particles = SplitData(*(self.init_samples(self.n_particles, key)
                                    for key in keys))
        except TypeError:
            particles = SplitData(*(self.init_samples,)*self.num_groups)
            self.n_particles = len(self.init_samples)
        self.d = particles.training.shape[1]
        return particles

    def get_params(self):
        return self.particles

    @partial(jit, static_argnums=0)
    def perturb(self, key, particles):
        """Add Gaussian noise scaled proportionally to the learning rate, that is
        noise = sqrt(2 * lr) * eps, where eps is distributed as a standard
        normal.
        Returns:
            pertubed particles"""
        assert self.optimizer_str == "sgd"
        keys = random.split(key, len(jax.tree_leaves(particles)))
        key_tree = jax.tree_unflatten(jax.tree_structure(particles), keys)
        scale = np.sqrt(2*self.learning_rate + 1e-8) * self.noise_level
        return jax.tree_multimap(partial(utils.add_gauss, scale=scale),
                                 key_tree,
                                 particles)

    def next_batch(self, key, batch_size=None): # TODO make this a generator or something
        """
        Return next subsampled batch of particles (training and validation) for the
        training of a gradient field approximator."""
        assert self.num_groups <= 2 # not strictly necessary anymore, but keep cause I'm not
        # expecting to use more groups
        particles, *_ = self.get_params()
        # subsample batch
        if batch_size is None:
            batch_size = self.n_particles//2

        # perturb
        if self.noise_level > 0:
            particles = self.perturb(key, particles)

        shuffled_batch = random.permutation(key, particles)
        return shuffled_batch[:batch_size], shuffled_batch[batch_size:]


    @partial(jit, static_argnums=0)
    def _step(self, key, particles, optimizer_state, params):
        """
        Updates particles in direction of the gradient.
        Arguments:
            params: can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or None.

        Returns:
            particles (updated)
            optimizer_state (updated)
            grad_aux: dict containing auxdata
        """
        key1, *keys = random.split(key, 3)
        particles = self.perturb(key1, particles)
        out = [self.gradient(params, k, p, aux=True) for p, k in zip(particles, keys)]
        grads, grad_aux = [SplitData(*o) for o in zip(*out)]
        grad_aux = {grouplabel + "_" + label: v
                    for grouplabel, d in grad_aux.items()
                    for label, v in d.items()}
        updated_grads, optimizer_state = self.opt.update(grads, optimizer_state, particles)
        particles = optax.apply_updates(particles, updated_grads)
        grad_aux.update({"grads": updated_grads})
        return particles, optimizer_state, grad_aux

    def step(self, params, key=None):
        """Log rundata, take step. Mutates state"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        updated_particles, self.optimizer_state, auxdata = self._step(
            key, self.particles, self.optimizer_state, params)
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

        if self.d < 300:
            auxdata.update({
                "step": step,
                "particles": particles,
    #            "gradient_norm": np.linalg.norm(gradient),
    #            "max_grad": np.max(np.abs(gradient)),
    #            "mean_grad": np.mean(np.abs(gradient)),
    #            "median_grad": np.median(np.abs(gradient)),
            })
        for k, v in particles.items():
            auxdata.update({
                f"{k}_mean": np.mean(v, axis=0),
                f"{k}_std":  np.std(v, axis=0),
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

    def plot_trajectories(self, ax=None, idx=None, **kwargs):
        if idx is None:
            idx = np.arange(self.n_particles*self.num_groups)
        if ax is None:
            ax = plt.gca()
        p_over_time = np.array(self.rundata["particles"])
        ax.plot(p_over_time[:, idx, 0], **kwargs)
        return ax

    def plot_final(self, target, ax=None, xlim=None, idx=None, **kwargs):
        if idx is None:
            idx = np.arange(self.n_particles*self.num_groups)
        if ax is None:
            ax = plt.gca()
        p = self.get_params()[idx]
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

    def animate_trajectory(self, target=None, fig=None, ax=None, interval=100, idx=None, **kwargs):
        """Create animated scatterplot of particle trajectory"""
        if idx is None:
            idx = np.arange(self.n_particles*self.num_groups)
        trajectory = np.asarray(self.rundata["particles"])[:, idx, :]
        if target is not None:
            plot.plot_fun_2d(target.pdf, lims=(-20, 20), ax=ax)
        anim = plot.animate_array(trajectory, fig, ax, interval=interval)
        return anim


class VectorFieldMixin:
    """Methods for init of Vector field MLP"""
    def __init__(self,
                 target_logp: callable,
                 target_dim: int,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 **kwargs):
        self.d = target_dim
        self.sizes = sizes if sizes else [32, 32, self.d]
        if self.sizes[-1] != self.d:
            warnings.warn(f"Output dim should equal target dim; instead "
                          f"received output dim {sizes[-1]} and "
                          f"target dim {self.d}.")

        self.threadkey, subkey = random.split(key)
        self.target_logp = target_logp

        # net and optimizer
        self.field = hk.transform(lambda x: nets.VectorField(self.sizes)(x))
        self.params = self.init_params()
        super().__init__(**kwargs)

    def init_params(self, key=None, keep_params=False):
        """Initialize MLP parameter"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.d)
        params = self.field.init(key, x_dummy)
        return params

    def get_params(self):
        return self.params

    def get_field(self, init_particles, params=None):
        """Retuns function v. v is a vector field, can take either single
        particle of shape (d,) or batch shaped (..., d)."""
        if params is None:
            params = self.get_params()
        norm = nets.get_norm(init_particles)
        #norm = lambda x: x
        def v(x):
            """x should have shape (n, d) or (d,)"""
            return self.field.apply(params, None, norm(x))
        return v


class TrainingMixin:
    """
    Methods for training NNs.
    Needs existence of a self.params at initialization.
    Methods to implement:
    * self.loss_fn
    * self._log
    """
    def __init__(self,
                 learning_rate: float = 1e-2,
                 patience: int = 10,
                 **kwargs):
        self.opt = optax.adam(learning_rate)
#        schedule = optax.polynomial_schedule(init_value=-learning_rate,
#                                             end_value=-learning_rate/100,
#                                             power=0.2,
#                                             transition_steps=200)
#        self.opt = optax.chain(optax.scale_by_adam(),
#                               optax.scale_by_schedule(schedule))
        self.optimizer_state = self.opt.init(self.params)

        # state and logging
        self.step_counter = 0
        self.rundata = {"train_steps": []}
        self.frozen_states = []
        self.patience = Patience(patience)
        super().__init__(**kwargs)

    @partial(jit, static_argnums=0)
    def _step(self, key, params, optimizer_state, particles, validation_particles):
        """update parameters and compute validation loss"""
        [loss, loss_aux], grads = value_and_grad(self.loss_fn, has_aux=True)(params, key, particles)
        grads, optimizer_state = self.opt.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, grads)

        _, val_loss_aux = self.loss_fn(params, key, validation_particles)
        auxdata = (loss_aux, val_loss_aux, grads, params)
        return params, optimizer_state, auxdata

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
        """
        Arguments
        * aux: list (train_aux, val_aux, grads, params)
        """
        raise NotImplementedError()

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        metrics.append_to_log(self.rundata, step_data)

    def train(self, next_batch: callable, key=None, n_steps=5, progress_bar=False):
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
            self.write_to_log({"model_params": self.get_params()})
            if self.patience.out_of_patience():
                self.patience.reset()
                break
        self.write_to_log({"train_steps": i+1})
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

    def loss_fn(self, params, key, particles):
        raise NotImplementedError()

    def gradient(self, params, key, particles, aux=False):
        raise NotImplementedError()


class SDLearner(VectorFieldMixin, TrainingMixin):
    """Parametrize vector field to maximize the stein discrepancy"""
    def __init__(self,
                 target_logp: callable,
                 target_dim: int,
                 key: np.array = random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 1e-2,
                 patience: int = 0):
        super().__init__(target_logp, target_dim, key=key, sizes=sizes,
                         learning_rate=learning_rate, patience=patience)
        self.lambda_reg = 1/2

    def loss_fn(self, params, key, particles):
        f = utils.negative(self.get_field(particles, params))
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            particles, self.target_logp, f, aux=True)
        l2_f_sq = utils.l2_norm_squared(particles, f)
        loss = -stein_discrepancy + self.lambda_reg * l2_f_sq
        aux = [loss, stein_discrepancy, l2_f_sq, stein_aux]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        """
        Arguments
        * aux: list (train_aux, val_aux, grads, params)
        """
        train_aux, val_aux, g, params = aux
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
                particles, self.target_logp, v, aux=True)
            auxdata = {"sd": stein_discrepancy}
            return v(particles), auxdata
        else:
            return v(particles)

    def grads(self, particles):
        """Same as `self.gradient` but uses state"""
        return self.gradient(self.get_params(), None, particles)


class KernelLearner(TrainingMixin):
    """Parametrize kernel to learn the KSD"""
    def __init__(self,
                 target_logp: callable,
                 target_dim: int,
                 key: np.array = random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 1e-2,
                 patience: int = 10):
        self.target_logp = target_logp
        self.lambda_reg = 1/2
        self.threadkey, key = random.split(key)
        self.num_leaders = None # first num_leaders of training particles used to find RKHS fit
        self.d = target_dim
        self.sizes = sizes
        if sizes:
            self.kernel = hk.transform(lambda x: nets.DeepKernel(sizes)(x))
        else:
            self.kernel = hk.transform(lambda x: nets.RBFKernel(
                scale_param=True, parametrization="log_diagonal")(x))
        self.params = self.init_params(key)
        super().__init__(learning_rate, patience)

    def init_params(self, key=None, keep_params=False):
        """Initialize kernel parameters"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.d)
        params = self.kernel.init(key, np.stack([x_dummy, x_dummy]))
        return params

    def get_params(self):
        return self.params

    def get_kernel_fn(self, init_particles, params=None):
        """
        Arguments:
        init_particles: samples from current dist q, shaped (n, d). Used
            to compute mean and stddev for normalziation.
        """
        if params is None:
            params = self.get_params()
        if self.sizes:
            norm = nets.get_norm(init_particles)
        else:
            norm = lambda x: x
        def kernel(x, y):
            return self.kernel.apply(params, None, np.stack([norm(x), norm(y)]))
        return kernel

    def get_field(self, inducing_particles, params=None):
        """return -phistar(\cdot)"""
        if params is None:
            params = self.get_params()
        kernel = self.get_kernel_fn(inducing_particles, params)
        def phistar(x):
            return stein.phistar_i(x, inducing_particles, self.target_logp, kernel, aux=False)
        return utils.negative(phistar)

    def loss_fn(self, params, key, particles):
        # *params, leaders = params # keep inducing particles fixed
        particles = random.permutation(key, particles)
        leaders = particles[:self.num_leaders]
        phi = utils.negative(self.get_field(leaders, params=params))
        #ksd_squared = stein.stein_discrepancy(particles, self.target_logp, phi)
        kernel = self.get_kernel_fn(leaders, params=params)
        #ksd_squared = stein.ksd_squared_l(particles, self.target_logp, kernel)
        ksd_squared = stein.ksd_squared_u(particles, self.target_logp, kernel)
        l2_squared = utils.l2_norm_squared(particles, phi)
        loss = -ksd_squared + self.lambda_reg * l2_squared
        aux = [loss, ksd_squared, l2_squared]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        """
        Arguments
        * aux: list (train_aux, val_aux, grads, params)
        """
        names = "loss ksd l2_norm".split()

        permuted_particles = random.permutation(random.PRNGKey(0), particles)
        kernel = self.get_kernel_fn(particles, aux[-1])
        mean_k = np.mean(vmap(kernel)(particles, permuted_particles))
        step_log = {"step_counter": step_counter,
                    "mean_k": mean_k}
        for group, data in zip(["training", "validation"], aux[:-2]):
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


class ScoreLearner(VectorFieldMixin, TrainingMixin):
    """Neural score matching"""
    def __init__(self,
                 target_logp,
                 target_dim,
                 key: np.array = random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 1e-2,
                 patience: int = 10,
                 lam: float = 0.):
        super().__init__(target_logp, target_dim, key=key, sizes=sizes,
                         learning_rate=learning_rate, patience=patience)
        self.lam=lam
        self.get_score = self.get_field
        self.lambda_reg = 1/2

    def loss_fn(self, params, key, particles):
        """Evaluates E[div(score)] + lambda * l2_norm(score)^2"""
        score = self.get_score(particles, params=params)
        mean_divergence  = np.mean(vmap(utils.div(score))(particles))
        mean_squared_div = np.mean(vmap(utils.div_sq(score))(particles))
        l2_norm_sq = utils.l2_norm_squared(particles, score)
        loss = mean_divergence + self.lambda_reg * l2_norm_sq + self.lam * mean_squared_div
        aux = [loss, mean_divergence, self.lambda_reg * l2_norm_sq, self.lam * mean_squared_div]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        """
        Arguments
        * aux: list (train_aux, val_aux, grads, params)
        """
        *loss_aux, g, params = aux
        layer_norms = [np.linalg.norm(v) for v in tree_util.tree_leaves(g)]
        step_log = {
            "step_counter": step_counter,
            "gradient_norms": layer_norms,
        }
        aux_names = "loss div l2_norm lam_term".split() # corresponds to elements in loss_aux
        for group, data in zip(["training", "validation"], loss_aux):
            step_log.update({
                group+"_"+name: value for name, value in zip(aux_names, data)
            })
        return step_log

    def gradient(self, params, _, particles, aux=False):
        score = self.get_score(init_particles=particles, params=params)
        grads = score(particles) - vmap(grad(self.target_logp))(particles)
        if aux:
            return grads, {}
        else:
            return grads

    def grads(self, particles):
        """Same as `self.gradient` but uses state"""
        return self.gradient(self.get_params(), None, particles)

class KernelGradient():
    """Computes the SVGD approximation to grad(KL), ie
    phi*(y) = E[grad(log p)(y) k(x, y) + div(k)(x, y)]"""
    def __init__(self,
                 target_logp,
                 key=random.PRNGKey(42),
                 kernel = kernels.get_rbf_kernel,
                 bandwidth=None,
                 lambda_reg = 1/2):
        self.target_logp = target_logp
        self.threadkey, subkey = random.split(key)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.rundata = {}

    def get_field(self, inducing_particles):
        """return -phistar(\cdot)"""
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, self.target_logp, inducing_particles)
        return utils.negative(phi), bandwidth

    def gradient(self, params, key, particles, aux=False, scaled=False):
        """Compute approximate KL gradient.
        params and key args are not used.
        particles is an np.ndarray of shape (n, d)"""
        v, h = self.get_field_scaled(particles) if scaled else self.get_field(particles)
        if aux:
            return vmap(v)(particles), {"bandwidth": h,
                                        "logp": vmap(self.target_logp)(particles)}
        else:
            return vmap(v)(particles)

    def get_field_scaled(self, inducing_particles):
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, self.target_logp, inducing_particles)
        l2_phi_squared = utils.l2_norm_squared(inducing_particles, phi)
        ksd = stein.stein_discrepancy(inducing_particles, self.target_logp, phi)
        alpha = ksd / (2*self.lambda_reg*l2_phi_squared)
        return utils.mul(phi, -alpha), bandwidth


class KernelizedScoreMatcher():
    """Compute kernelized score estimate"""
    def __init__(self,
                target_logp: callable,
                key=random.PRNGKey(42),
                kernel = kernels.get_rbf_kernel(1),
                lambda_reg=1/2,
                scale=1.):
        self.target_logp = target_logp
        self.threadkey, subkey = random.split(key)
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.scale=scale
        self.rundata = {}

    def objective(self, particles, f):
        divergence_term = -np.mean(vmap(utils.div(f))(particles))
        return divergence_term

    def target_score(self, x):
        return grad(self.target_logp)(x) / (2*self.lambda_reg)

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
                return grad(self.target_logp)(y) * self.kernel(y, x_)
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
                target_logp,
                key=random.PRNGKey(42),
                lambda_reg=1/2):
        self.target_logp = target_logp
        self.threadkey, subkey = random.split(key)
        self.lambda_reg = lambda_reg
        self.rundata = {}

    def target_score(self, x):
        return grad(self.target_logp)(x) / (2*self.lambda_reg)

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
