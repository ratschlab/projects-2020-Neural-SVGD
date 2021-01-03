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
    """Train-test split"""
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
                 learning_rate=1e-2,
                 optimizer="sgd",
                 custom_optimizer = None,
                 num_groups=1,
                 n_particles: int = 50,
                 compute_metrics = None):
        """
        Arguments
        ----------
        gradient: takes in args (params, key, particles) and returns
            an array of shape (n, d), interpreted as grad(loss)(particles).
        init_samples: either a callable sample(num_samples, key), or an nd.array
        of shape (n, d) containing initial samples.
        compute_metrics: callable, takes in particles as array of shape (n, d) and
        outputs a dict shaped {'name': metric for name, metric in zip(names, metrics)}.
        Evaluated once every 50 steps.
        """
        self.gradient = gradient
        self.n_particles = n_particles
        self.num_groups = num_groups
        self.threadkey, subkey = random.split(key)
        self.init_samples = init_samples
        self.particles = self.init_particles(subkey)

        # optimizer for particle updates
        if custom_optimizer:
            self.optimizer_str = "custom"
            self.learning_rate = None
            self.opt = custom_optimizer
        else:
            self.optimizer_str = optimizer
            self.learning_rate = learning_rate
            self.opt = utils.optimizer_mapping[optimizer](learning_rate)
        self.optimizer_state = self.opt.init(self.particles)
        self.step_counter = 0
        self.rundata = {}
        self.donedone = False
        self.compute_metrics = compute_metrics

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

    def next_batch(self, key, batch_size=None): # TODO make this a generator or something
        """
        Return next subsampled batch of training particles (split into training
        and validation) for the training of a gradient field approximator."""
        particles, *_ = self.get_params()

        # subsample batch
        if batch_size is None:
            batch_size = 3*self.n_particles//4

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
        out = [self.gradient(params, p, aux=True) for p in particles]
        grads, grad_aux = [SplitData(*o) for o in zip(*out)]
        grad_aux = {grouplabel + "_" + label: v
                    for grouplabel, d in grad_aux.items()
                    for label, v in d.items()}
        updated_grads, optimizer_state = self.opt.update(grads, optimizer_state, particles)
        particles = optax.apply_updates(particles, updated_grads)
        #grad_aux.update({"grads": updated_grads})
        return particles, optimizer_state, grad_aux

    def step(self, params, key=None):
        """Log rundata, take step. Mutates state"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        updated_particles, self.optimizer_state, auxdata = self._step(
            key, self.particles, self.optimizer_state, params)
        self.log(auxdata)
        self.particles = updated_particles
        self.step_counter += 1
        return None

    def log(self, grad_aux=None):
        metrics.append_to_log(self.rundata, self._log(self.particles, self.step_counter))
        if self.step_counter % 10 == 0 and self.compute_metrics:
            aux_metrics = self.compute_metrics(self.particles.training)
            metrics.append_to_log(self.rundata,
                                  {k: (self.step_counter, v) for k, v in aux_metrics.items()})
        if grad_aux is not None:
            metrics.append_to_log(self.rundata, grad_aux)


    @partial(jit, static_argnums=0)
    def _log(self, particles, step):

        auxdata = {}
        if self.d < 35:
            auxdata.update({
                "step": step,
                "particles": particles,
            })
        for k, v in particles.items():
            auxdata.update({
                f"{k}_mean": np.mean(v, axis=0),
                f"{k}_std":  np.std(v, axis=0),
            })
        return auxdata

    def done(self):
        """converts rundata into arrays"""
        if self.donedone:
            print("already done.")
            return
        skip = "particles accuracy test_logp".split()
        self.rundata = {
            k: v if k in skip else np.array(v)
            for k, v in self.rundata.items()
        }
        if "particles" in self.rundata:
            d = SplitData(*[np.array(trajectory)
                        for trajectory in zip(*self.rundata["particles"])])
            self.rundata["particles"] = d
        self.donedone = True


    def plot_mean_and_std(self, target=None, axs=None, **kwargs):
        """axs: two axes"""
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
                 target_dim: int,
                 target_logp: callable = None,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 aux = True,
                 **kwargs):
        """if aux, then add mean and variance as auxiliary input to MLP."""
        self.aux = aux
        self.d = target_dim
        self.sizes = sizes if sizes else [32, 32, self.d]
        self.auxdim = self.d*2
        if self.sizes[-1] != self.d:
            warnings.warn(f"Output dim should equal target dim; instead "
                          f"received output dim {sizes[-1]} and "
                          f"target dim {self.d}.")
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        def field(x, aux):
            mlp = nets.MLP(self.sizes)
            scale = hk.get_parameter("scale", (), init=lambda *args: np.ones(*args))
            mlp_input = np.concatenate([x, aux]) if self.aux else x
            return scale * mlp(mlp_input)
        self.field = hk.transform(field)
        self.params = self.init_params()
        super().__init__(**kwargs)

    def compute_aux(self, particles):
        """Auxiliary data that will be concatenated onto MLP input.
        Output has shape (self.auxdim,).
        Can also be None."""
        if not self.aux:
            return None
        aux = np.concatenate([np.mean(particles, axis=0), np.std(particles, axis=0)])
        assert self.auxdim == len(aux)
        return aux

    def init_params(self, key=None, keep_params=False):
        """Initialize MLP parameter"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.d)
        aux_dummy = np.ones(self.auxdim) if self.aux else None
        params = self.field.init(key, x_dummy, aux_dummy)
        return params

    def get_params(self):
        return self.params

    def get_field(self, init_particles, params=None):
        """Retuns function v. v is a vector field, can take either single
        particle of shape (d,) or batch shaped (..., d)."""
        if params is None:
            params = self.get_params()
        norm = nets.get_norm(init_particles)
        aux = self.compute_aux(init_particles)
        #norm = lambda x: x
        def v(x):
            """x should have shape (n, d) or (d,)"""
            return self.field.apply(params, None, norm(x), aux)
        return v


class EBMMixin():
    def __init__(self,
                 target_dim: int,
                 target_logp: callable = None,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 **kwargs):
        self.d = target_dim
        self.sizes = sizes if sizes else [32, 32, 1]
        if self.sizes[-1] != 1:
            warnings.warn(f"Output dim should equal 1; instead "
                          f"received output dim {sizes[-1]}")
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        self.ebm = hk.transform(
            lambda *args: nets.MLP(self.sizes)(*args))
        self.params = self.init_params()
        super().__init__(**kwargs)

    def init_params(self, key=None):
        """Initialize MLP parameter"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.d)
        params = self.ebm.init(key, x_dummy)
        return params

    def get_params(self):
        return self.params

    def get_field(self, init_particles, params=None):
        if params is None:
            params = self.get_params()
        #norm = nets.get_norm(init_particles)
        norm = lambda x: x
        def ebm(x):
            """x should have shape (d,)"""
            return np.squeeze(self.ebm.apply(params, None, norm(x)))
        return grad(ebm)


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
    def _step(self, key, params, batch, optimizer_state, particles, validation_particles):
        """update parameters and compute validation loss
        batch: batch of data used to approximate logp"""
        [loss, loss_aux], grads = value_and_grad(self.loss_fn, has_aux=True)(params, batch, key, particles)
        grads, optimizer_state = self.opt.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, grads)

        _, val_loss_aux = self.loss_fn(params, batch, key, validation_particles)
        auxdata = (loss_aux, val_loss_aux, grads, params)
        return params, optimizer_state, auxdata

    def step(self, particles, validation_particles, batch):
        """Step and mutate state"""
        self.threadkey, key = random.split(self.threadkey)
        self.params, self.optimizer_state, auxdata = self._step(
            key, self.params, batch, self.optimizer_state,
            particles, validation_particles)
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

    def train(self, batch = None, next_batch: callable = None, key=None, n_steps=5, progress_bar=False, data = None, early_stopping = True):
        """
        batch and next_batch cannot both be None.
        batch is an array of particles.
        data_batch is data used to compute logp (passed through self.loss_fn)

        Arguments:
        * batch: arrays (training, validation) of particles, shaped (n, d) resp (m, d)
        * next_batch: callable, outputs next training batch. Signature:
            next_batch(key)
        """
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def step(key):
            train_x, val_x = next_batch(key) if next_batch else batch
            self.step(train_x, val_x, data)
            val_loss = self.rundata["validation_loss"][-1]
            self.patience.update(val_loss)
            return

        for i in tqdm(range(n_steps), disable=not progress_bar):
            key, subkey = random.split(key)
            step(subkey)
            #self.write_to_log({"model_params": self.get_params()})
            if self.patience.out_of_patience() and early_stopping:
                self.patience.reset()
                break
        self.write_to_log({"train_steps": i+1})
        return

    def freeze_state(self):
        """Stores current state as tuple (step_counter, params, rundata)"""
        self.frozen_states.append((self.step_counter,
                                   self.get_params(),
                                   self.rundata))
        return

    def loss_fn(self, params, batch, key, particles):
        raise NotImplementedError()

    def gradient(self, params, particles, aux=False):
        raise NotImplementedError()


class SDLearner(VectorFieldMixin, TrainingMixin):
    """Parametrize vector field to maximize the stein discrepancy"""
    def __init__(self,
                 target_dim: int,
                 target_logp: callable = None,
                 get_target_logp: callable = None,
                 key: np.array = random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 5e-3,
                 patience: int = 0,
                 aux=True,
                 lambda_reg=1/2):
        """aux: bool, whether to concatenate particle dist info onto
        mlp input"""
        super().__init__(target_dim, target_logp, key=key, sizes=sizes,
                         learning_rate=learning_rate, patience=patience, aux=aux)
        self.lambda_reg = lambda_reg
        if target_logp:
            assert not get_target_logp
            self.get_target_logp = lambda *args: target_logp
        elif get_target_logp:
            self.get_target_logp = get_target_logp
        else:
            return ValueError(f"One of target_logp and get_target_logp must"
                              f"be given.")
        self.scale = 1. # scaling of self.field

    def loss_fn(self, params, batch, key, particles):
        """
        params: neural net paramers
        batch: data used to compute logp. Can be none if logp is known precisely
        key: random PRNGKey
        particles: array of shape (n, d)
        """
        target_logp = self.get_target_logp(batch)
        f = utils.negative(self.get_field(particles, params))
        #f = utils.negative(self._get_grad(particles, params)) # = - grad(KL)
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            particles, target_logp, f, aux=True)
        l2_f_sq = utils.l2_norm_squared(particles, f)
        loss = -stein_discrepancy + self.lambda_reg * l2_f_sq
        #loss = - 1/2 * stein_discrepancy**2 / l2_f_sq
        aux = [loss, stein_discrepancy, l2_f_sq, stein_aux]
        return loss, aux

    def _loss_fn(self, params, batch, key, particles):
        target_logp = self.get_target_logp(batch)
        f = utils.negative(self.get_field(particles, params))
        stein_discrepancy, stein_aux = stein.stein_discrepancy(
            particles, target_logp, f, aux=True)
        l2_f_sq = utils.l2_norm_squared(particles, f)
        sd = stein_discrepancy**2 / l2_f_sq
        loss = -sd
        aux = [loss, sd, l2_f_sq, stein_aux]
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

    def gradient(self, params, particles, aux=False):
        """params is a pytree of neural net parameters"""
        v = vmap(self.get_field(particles, params))
        #v = vmap(self._get_grad(particles, params))
        if aux:
            return v(particles), {}
        else:
            return v(particles)

    def grads(self, particles):
        """Same as `self.gradient` but uses state"""
        return self.gradient(self.get_params(), particles)

    def done(self):
        """converts rundata into arrays"""
        self.rundata = {
            k: v if k in ["model_params", "gradient_norms"] else np.array(v)
            for k, v in self.rundata.items()
        }


class KernelGradient():
    """Computes the SVGD approximation to grad(KL), ie
    phi*(y) = E[grad(log p)(y) k(x, y) + div(k)(x, y)]"""
    def __init__(self,
                 target_logp: callable = None,
                 get_target_logp: callable = None,
                 kernel = kernels.get_rbf_kernel,
                 bandwidth=None,
                 scaled=False,
                 lambda_reg = 1/2):
        """get_target_log is a callable that takes in a batch of data
        (can be any pytree of jnp.ndarrays) and returns a callable logp
        that computes the target log prob (up to an additive constant).
        scaled: whether to rescale gradients st. they match
        (grad(logp) - grad(logp))/(2 * lambda_reg) in scale
        """
        if target_logp:
            assert not get_target_logp
            self.get_target_logp = lambda *args: target_logp
        elif get_target_logp:
            self.get_target_logp = get_target_logp
        else:
            return ValueError(f"One of target_logp and get_target_logp must"
                              f"be given.")
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.rundata = {}
        self.scaled = scaled

    def get_field(self, inducing_particles, batch=None):
        """return -phistar(\cdot)"""
        target_logp = self.get_target_logp(batch)
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, target_logp, inducing_particles)
        return utils.negative(phi), bandwidth

    def gradient(self, batch, particles, aux=False):
        """Compute approximate KL gradient.
        particles is an np.ndarray of shape (n, d)"""
        target_logp = self.get_target_logp(batch)
        v, h = self.get_field_scaled(particles, batch) if self.scaled \
            else self.get_field(particles, batch)
        if aux:
            return vmap(v)(particles), {"bandwidth": h,
                                        "logp": vmap(target_logp)(particles)}
        else:
            return vmap(v)(particles)

    def get_field_scaled(self, inducing_particles, batch=None):
        target_logp = self.get_target_logp(batch)
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, target_logp, inducing_particles)
        l2_phi_squared = utils.l2_norm_squared(inducing_particles, phi)
        ksd = stein.stein_discrepancy(inducing_particles, target_logp, phi)
        alpha = ksd / (2*self.lambda_reg*l2_phi_squared)
        return utils.mul(phi, -alpha), bandwidth


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

    def gradient(self, _, particles, aux=False):
        """Compute gradient used for SGD particle update"""
        v = self.get_field(particles)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)
