import jax.numpy as np
from jax import jit, vmap, random, value_and_grad, grad
import haiku as hk
import jax
import optax

from tqdm import tqdm
from functools import partial
import warnings

import utils
import metrics
import stein
import kernels
import nets

from typing import Mapping
import os
on_cluster = not os.getenv("HOME") == "/home/lauro"
disable_tqdm = on_cluster

"""
This file implements methods that simulate different kinds of particle dynamics.
It is structured as follows:

The class `Particles` acts as container for the particle positions and associated data.
Any update rule can be 'plugged in' by supplying the `gradient` argument. The following
update rules are implemented here:
- `SDLearner`: the method developed in this project, which dynamically learns a trajectory using a neural network.
- `KernelGradient`: simulates SVGD dynamics
- `EnergyGradient`: simulates Langevin dynamics

Finally, the mixin classes `VectorFieldMixin` and `EBMMixin` define different constraints on the neural update rule.
"""


class Patience:
    """Criterion for early stopping"""
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.time_waiting = 0
        self.min_validation_loss = None
        self.disable = patience == -1

    def update(self, validation_loss):
        """Returns True when early stopping criterion (validation loss
        failed to decrease for `self.patience` steps) is met"""
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
                 custom_optimizer=None,
                 n_particles: int = 50,
                 compute_metrics=None):
        """
        Args:
            gradient: takes in args (params, key, particles) and returns
        an array of shape (n, d). Used to compute particle update x = x + eps * gradient(*args)
            init_samples: either a callable sample(num_samples, key), or an array
        of shape (n, d) containing initial samples.
            learning_rate: scalar step-size for particle updates
            compute_metrics: callable, takes in particles as array of shape (n, d) and
        outputs a dict shaped {'name': metric for name, metric in
        zip(names, metrics)}. Evaluated once every 50 steps.
        """
        self.gradient = gradient
        self.n_particles = n_particles
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
        """Returns an jnp.ndarray of shape (n, d) containing particles."""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        if callable(self.init_samples):
            particles = self.init_samples(self.n_particles, key)
        else:
            particles = self.init_samples
            self.n_particles = len(particles)
        self.d = particles.shape[1]
        return particles

    def get_params(self):
        return self.particles

    def next_batch(self,
                   key,
                   n_train_particles: int = None,
                   n_val_particles: int = None):
        """
        Return next subsampled batch of training particles (split into training
        and validation) for the training of a gradient field approximator.
        """
        particles = self.get_params()
        shuffled_batch = random.permutation(key, particles)

        if n_train_particles is None:
            if n_val_particles is None:
                n_val_particles = self.n_particles // 4
            n_train_particles = self.n_particles - n_val_particles
        elif n_val_particles is None:
            n_val_particles = self.n_particles - n_train_particles

        assert n_train_particles + n_val_particles == self.n_particles
        return shuffled_batch[:n_train_particles], shuffled_batch[-n_val_particles:]

    @partial(jit, static_argnums=0)
    def _step(self, key, particles, optimizer_state, params):
        """
        Updates particles in the direction given by self.gradient

        Arguments:
            particles: jnp.ndarray of shape (n, d)
            params: can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or None.

        Returns:
            particles (updated)
            optimizer_state (updated)
            grad_aux: dict containing auxdata
        """
        grads, grad_aux = self.gradient(params, particles, aux=True)
        updated_grads, optimizer_state = self.opt.update(grads, optimizer_state, particles)
        particles = optax.apply_updates(particles, updated_grads)
        grad_aux.update({
            "global_grad_norm": optax.global_norm(grads),
            "global_grad_norm_post_update": optax.global_norm(updated_grads),
        })
        grad_aux.update({})
        # grad_aux.update({"grads": updated_grads})
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
            aux_metrics = self.compute_metrics(self.particles)
            metrics.append_to_log(self.rundata,
                                  {k: (self.step_counter, v) for k, v in aux_metrics.items()})
        if grad_aux is not None:
            metrics.append_to_log(self.rundata, grad_aux)

    @partial(jit, static_argnums=0)
    def _log(self, particles, step):
        auxdata = {}
        if self.d < 400:
            auxdata.update({
                "step": step,
                "particles": particles,
                "mean": np.mean(particles, axis=0),
                "std": np.std(particles, axis=0),
            })
        return auxdata

    def done(self):
        """converts rundata into arrays"""
        if self.donedone:
            print("already done.")
            return
        skip = "particles accuracy".split()
        self.rundata = {
            k: v if k in skip else np.array(v)
            for k, v in self.rundata.items()
        }
        if "particles" in self.rundata:
            self.rundata["particles"] = np.array(self.rundata['particles'])
        self.donedone = True


class VectorFieldMixin:
    """Methods for init of vector field MLP"""
    def __init__(self,
                 target_dim: int,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 aux=False,
                 normalize_inputs=False,
                 **kwargs):
        """
        args:
            aux: bool; whether to add mean and std as auxiliary input to MLP.
            normalize_inputs: whether to normalize particles
        """
        self.aux = aux
        self.d = target_dim
        self.sizes = sizes if sizes else [32, 32, self.d]
        self.auxdim = self.d*2
        if self.sizes[-1] != self.d:
            warnings.warn(f"Output dim should equal target dim; instead "
                          f"received output dim {sizes[-1]} and "
                          f"target dim {self.d}.")
        self.threadkey, subkey = random.split(key)
        self.normalize_inputs = normalize_inputs

        # net and optimizer
        def field(x, aux, dropout: bool = False):
            mlp = nets.MLP(self.sizes)
            scale = hk.get_parameter("scale", (), init=lambda *args: np.ones(*args))
            mlp_input = np.concatenate([x, aux]) if self.aux else x
            return scale * mlp(mlp_input, dropout)
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

    def get_field(self, key, init_particles, params=None, dropout=False):
        """Retuns function v. v is a vector field, can take either single
        particle of shape (d,) or batch shaped (..., d)."""
        if params is None:
            params = self.get_params()
        if self.normalize_inputs:
            norm = nets.get_norm(init_particles)
        else:
            norm = lambda x: x
        aux = self.compute_aux(init_particles)

        def v(x):
            """x should have shape (n, d) or (d,)"""
            return self.field.apply(params, key, norm(x), aux, dropout=dropout)
        return v


class EBMMixin():
    def __init__(self,
                 target_dim: int,
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

    def get_field(self, key, init_particles, params=None, dropout=False):
        del init_particles
        if params is None:
            params = self.get_params()

        def ebm(x):
            """x should have shape (d,)"""
            # norm = nets.get_norm(init_particles)
            # x = norm(x)
            return np.squeeze(self.ebm.apply(params, key, x))
        return grad(ebm)


class TrainingMixin:
    """
    Encapsulates methods for training the Stein network (which approximates
    the particle update). Agnostic re: architecture. Needs existence of 
    a self.params at initialization.
    Methods to implement:
    * self.loss_fn
    * self._log
    """
    def __init__(self,
                 learning_rate: float = 1e-2,
                 patience: int = 10,
                 dropout: bool = False,
                 **kwargs):
        """
        args:
        dropout: whether to use dropout during training
        """
#        schedule_fn = optax.piecewise_constant_schedule(
#                -learning_rate, {50: 1/5, 100: 1/2})
#        self.opt = optax.chain(
#                optax.scale_by_adam(),
#                optax.scale_by_schedule(schedule_fn))
        self.opt = optax.adam(learning_rate)
        self.optimizer_state = self.opt.init(self.params)
        self.dropout = dropout

        # state and logging
        self.step_counter = 0
        self.rundata = {"train_steps": []}
        self.frozen_states = []
        self.patience = Patience(patience)
        super().__init__(**kwargs)

    @partial(jit, static_argnums=0)
    def _step(self,
              key,
              params,
              optimizer_state,
              dlogp,
              val_dlogp,
              particles,
              validation_particles):
        """
        update parameters and compute validation loss
        args:
            dlogp: array of shape (n_train, d)
            val_dlogp: array of shape (n_validation, d)
        """
        [loss, loss_aux], grads = value_and_grad(self.loss_fn,
                                                 has_aux=True)(params,
                                                               dlogp,
                                                               key,
                                                               particles,
                                                               dropout=self.dropout)
        grads, optimizer_state = self.opt.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, grads)

        _, val_loss_aux = self.loss_fn(params,
                                       val_dlogp,
                                       key,
                                       validation_particles,
                                       dropout=False)
        auxdata = (loss_aux, val_loss_aux, grads, params)
        return params, optimizer_state, auxdata

    def step(self, particles, validation_particles, dlogp, val_dlogp):
        """Step and mutate state"""
        self.threadkey, key = random.split(self.threadkey)
        self.params, self.optimizer_state, auxdata = self._step(
            key, self.params, self.optimizer_state, dlogp, val_dlogp,
            particles, validation_particles)
        self.write_to_log(
            self._log(particles, validation_particles, auxdata, self.step_counter))
        self.step_counter += 1
        return None

    def _log(self, particles, val_particles, auxdata, step_counter):  # depends on loss_fn aux
        """
        Arguments
        * aux: list (train_aux, val_aux, grads, params)
        """
        raise NotImplementedError()

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        metrics.append_to_log(self.rundata, step_data)

    def train(self,
              split_particles,
              split_dlogp,
              n_steps=5,
              early_stopping=True,
              progress_bar=False):
        """
        batch and next_batch cannot both be None.

        Arguments:
            split_particles: arrays (training, validation) of particles,
                shaped (n, d) resp (m, d)
            split_dlogp: arrays (training, validation) of loglikelihood
                gradients. Same shape as split_particles.
            key: random.PRGNKey
            n_steps: int, nr of steps to train
        """
        self.patience.reset()

        def step():
            self.step(*split_particles, *split_dlogp)
            val_loss = self.rundata["validation_loss"][-1]
            self.patience.update(val_loss)
            return

        for i in tqdm(range(n_steps), disable=not progress_bar):
            step()
            # self.write_to_log({"model_params": self.get_params()})
            if self.patience.out_of_patience() and early_stopping:
                break
        self.write_to_log({"train_steps": i+1})
        return

    def warmup(self,
               key,
               sample_split_particles: callable,
               next_data: callable = lambda: None,
               n_iter: int = 10,
               n_inner_steps: int = 30,
               progress_bar: bool = False,
               early_stopping: bool = True):
        """resample from particle initializer to stabilize the beginning
        of the trajectory
        args:
            key: prngkey
            sample_split_particles: produces next x_train, x_val sample
            next_data: produces next batch of data
            n_iter: number of iterations (50 training steps each)
        """
        for _ in tqdm(range(n_iter), disable=not progress_bar):
            key, subkey = random.split(key)
            self.train(sample_split_particles(subkey),
                       n_steps=n_inner_steps,
                       data=next_data(),
                       early_stopping=early_stopping)

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
                 key: np.array = random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 5e-3,
                 patience: int = 0,
                 aux=True,
                 lambda_reg=1/2,
                 use_hutchinson: bool = False,
                 dropout=False,
                 normalize_inputs=False):
        """
        args:
            aux: bool, whether to concatenate particle dist info onto
        mlp input
            use_hutchinson: when True, use Hutchinson's estimator to
        compute the stein discrepancy.
            normalize_inputs: normalize particles
        """
        super().__init__(target_dim, key=key, sizes=sizes,
                         learning_rate=learning_rate, patience=patience,
                         aux=aux, dropout=dropout, normalize_inputs=normalize_inputs)
        self.lambda_reg = lambda_reg
        self.scale = 1.  # scaling of self.field
        self.use_hutchinson = use_hutchinson

    def loss_fn(self,
                params,
                dlogp: np.ndarray,
                key: np.ndarray,
                particles: np.ndarray,
                dropout: bool = False):
        """
        Arguments:
            params: neural net paramers
            dlogp: gradient grad(log p)(x), shaped (n, d)
            key: random PRNGKey
            particles: array of shape (n, d)
            dropout: whether to use dropout in the gradient network
        """
        key, subkey = random.split(key)
        f = utils.negative(self.get_field(subkey, particles, params, dropout=dropout))
        key, subkey = random.split(key)
        if self.use_hutchinson:
            stein_discrepancy = stein.stein_discrepancy_hutchinson_fixed_log(
                subkey, particles, dlogp, f)
            stein_aux = np.array([1, 1.])
        else:
            stein_discrepancy = stein.stein_discrepancy_fixed_log(
                particles, dlogp, f)
            stein_aux = np.array([1., 1.])
        l2_f_sq = utils.l2_norm_squared(particles, f)
        loss = -stein_discrepancy + self.lambda_reg * l2_f_sq  # + optax.global_norm(params)**2
        # loss = - 1/2 * stein_discrepancy**2 / l2_f_sq
        aux = [loss, stein_discrepancy, l2_f_sq, stein_aux]
        return loss, aux

    @partial(jit, static_argnums=0)
    def _log(self, particles, validation_particles, aux, step_counter):
        """
        Arguments:
            aux: list (train_aux, val_aux, grads, params)
        """
        train_aux, val_aux, g, params = aux
        loss, sd, l2v, stein_aux = train_aux
        drift, repulsion = stein_aux  # shape (2, d)
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
            "layer_gradient_norms": gradient_norms,
            "global_gradient_norm": optax.global_norm(g),
        }
        return step_log

    def gradient(self, params, particles, aux=False):
        """
        Plug-in particle update method. No dropout.
        args:
            params: pytree of neural net parameters
            particles: array of shape (n, d)
            aux: bool
        """
        v = vmap(self.get_field(None, particles, params, dropout=False))
        # v = vmap(self._get_grad(particles, params))
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
                 target_logp: callable = None,  # TODO replace with dlogp supplied as array
                 get_target_logp: callable = None,
                 kernel=kernels.get_rbf_kernel,
                 bandwidth=None,
                 scaled=False,
                 lambda_reg=1/2,
                 use_hutchinson: bool = False):
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
            return ValueError("One of target_logp and get_target_logp must"
                              "be given.")
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.rundata = {}
        self.scaled = scaled

    def get_field(self, inducing_particles, batch=None):
        """return -phistar"""
        target_logp = self.get_target_logp(batch)
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, target_logp, inducing_particles)
        return utils.negative(phi), bandwidth

    def gradient(self, batch, particles, aux=False):
        """Compute approximate KL gradient.
        args:
            batch: minibatch data used to estimate logp (can be None)
            particles: array of shape (n, d)
        """
        target_logp = self.get_target_logp(batch)
        v, h = self.get_field_scaled(particles, batch) if self.scaled \
            else self.get_field(particles, batch)
        if aux:
            return vmap(v)(particles), {"bandwidth": h,
                                        "logp": vmap(target_logp)(particles)}
        else:
            return vmap(v)(particles)

    def get_field_scaled(self, inducing_particles, batch=None):
        hardcoded_seed = random.PRNGKey(0)  # TODO seed should change across iters
        target_logp = self.get_target_logp(batch)
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, target_logp, inducing_particles)
        l2_phi_squared = utils.l2_norm_squared(inducing_particles, phi)
        if self.use_hutchinson:
            ksd = stein.stein_discrepancy_hutchinson(hardcoded_seed, inducing_particles, target_logp, phi)
        else:
            ksd = stein.stein_discrepancy(inducing_particles, target_logp, phi)
        alpha = ksd / (2*self.lambda_reg*l2_phi_squared)
        return utils.mul(phi, -alpha), bandwidth


class EnergyGradient():
    """Compute pure SGLD gradient grad(log p)(x) (without noise)"""
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
        """Return vector field used for updating, grad(log p)(x)$
        (without noise)."""
        return utils.negative(self.target_score)

    def gradient(self, _, particles, aux=False):
        """Compute gradient used for SGD particle update"""
        v = self.get_field(particles)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)
