import os
import argparse
from itertools import cycle
from datetime import datetime
import numpy as onp
import jax
from jax import numpy as jnp
from jax import vmap, pmap, random, jit
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import optax
from convnet import model, crossentropy_loss, log_prior, ensemble_accuracy
import models
import metrics
import config as cfg

on_cluster = not os.getenv("HOME") == "/home/lauro"

# cli args
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=100, help="Number of parallel chains")
parser.add_argument("--num_epochs", type=int, default=1)
args = parser.parse_args()

# Config
# date = datetime.today().strftime('%a-%H:%M-%f')
results_file = cfg.results_path + "bnn-nsvgd.csv"
BATCH_SIZE = 128
DISABLE_PROGRESS_BAR = on_cluster
USE_PMAP = False
LAMBDA_REG = 10**2
LAYER_SIZE = 128 if on_cluster else 32

if USE_PMAP:
    vpmap = pmap
else:
    vpmap = vmap

print("Loading data...")
# Load MNIST
data_dir = './data' if on_cluster else '/tmp/tfds'
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']

# Full train and test set
train_images, train_labels = train_data['image'], train_data['label']
test_images, test_labels = test_data['image'], test_data['label']

# Split off the validation set
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=0)
data_size = len(train_images)



def make_batches(images, labels, batch_size, cyclic=True):
    """Returns an iterator through tuples (image_batch, label_batch).
    if cyclic, then the iterator cycles back after exhausting the batches"""
    num_batches = len(images) // batch_size
    split_idx = onp.arange(1, num_batches+1)*batch_size
    batches = zip(*[onp.split(data, split_idx, axis=0) for data in (images, labels)])
    return cycle(batches) if cyclic else batches


def loss(params, images, labels):
    """Minibatch approximation of the (unnormalized) Bayesian
    negative log-posterior evaluated at `params`. That is,
    -log model_likelihood(data_batch | params) * batch_rescaling_constant - log prior(params))"""
    logits = model.apply(params, images)
    return data_size/BATCH_SIZE * crossentropy_loss(logits, labels) - log_prior(params)


validation_batches = make_batches(val_images, val_labels, BATCH_SIZE, cyclic=False)

# utility functions for dealing with parameters
params_tree = model.init(random.PRNGKey(0), train_images[:2])
_, unravel = jax.flatten_util.ravel_pytree(params_tree)


def ravel(tree):
    return jax.flatten_util.ravel_pytree(tree)[0]


def init_flat_params(key):
    return ravel(model.init(key, train_images[:2]))


def get_minibatch_logp(batch):
    """
    Returns a callable that computes target posterior
    given flattened param vector.

    args:
        batch: tuple (images, labels)
    """
    def minibatch_logp(params_flat):
        return -loss(unravel(params_flat), *batch)
    return minibatch_logp


def sample_tv(key):
    """return two sets of particles at initialization, for
    training and validation in the warmup phase"""
    return vmap(init_flat_params)(random.split(key, args.num_samples)).split(2)


@jit
def minibatch_accuracy(param_set_flat, images, labels):
    return ensemble_accuracy(vmap(unravel)(param_set_flat), images, labels)


def compute_acc(param_set_flat):
    accs = []
    for batch in validation_batches:
        accs.append(minibatch_accuracy(param_set_flat, *batch))
    return onp.mean(accs)


def vmean(fun):
    """vmap, but computes mean along mapped axis"""
    def compute_mean(*args, **kwargs):
        return jnp.mean(vmap(fun)(*args, **kwargs), axis=-1)
    return compute_mean


def train(key,
          meta_lr: float = 1e-3,
          particle_stepsize: float = 1e-3,
          patience: int = 0,
          max_train_steps: int = 10,
          evaluate_every: int = 10,
          dropout: bool = True,
          write_results_to_file: bool = False):
    """
    Initialize model; warmup; training; evaluation.
    Returns a dictionary of metrics.
    Args:
        meta_lr: learning rate of Stein network
        particle_stepsize: learning rate of BNN
        patience: early stopping criterion
        max_train_steps: cutoff for Stein network training iteration
        evaluate_every: compute metrics every `evaluate_every` steps
        dropout: use dropout during training of the Stein network
        write_results_to_file: whether to save accuracy in csv file
    """
    # initialize particles and the dynamics model
    key, subkey = random.split(key)
    init_particles = vmap(init_flat_params)(random.split(subkey, args.num_samples))
    opt = optax.sgd(particle_stepsize)

    #opt = optax.chain(
    #    optax.scale_by_adam(),
    #    optax.scale(-particle_stepsize),
    #)

    key, subkey1, subkey2 = random.split(key, 3)
    neural_grad = models.SDLearner(target_dim=init_particles.shape[1],
                                   get_target_logp=get_minibatch_logp,
                                   learning_rate=meta_lr,
                                   key=subkey1,
                                   sizes=[LAYER_SIZE, LAYER_SIZE, LAYER_SIZE, init_particles.shape[1]],
                                   aux=False,
                                   use_hutchinson=True,
                                   lambda_reg=LAMBDA_REG,
                                   patience=patience,
                                   dropout=dropout)

    particles = models.Particles(key=subkey2,
                                 gradient=neural_grad.gradient,
                                 init_samples=init_particles,
                                 custom_optimizer=opt)

    train_batches = make_batches(train_images, train_labels, BATCH_SIZE)


    def step(key, train_batch):
        """one iteration of the particle trajectory simulation"""
        neural_grad.train(split_particles=particles.next_batch(key),
                          n_steps=max_train_steps,
                          data=train_batch)
        particles.step(neural_grad.get_params())
        return

    def evaluate(step_counter, ps):
        stepdata = {
            "accuracy": compute_acc(ps),
            "step_counter": step_counter,
        }
        if write_results_to_file:
            with open(results_file, "a") as file:
                file.write(f"{step_counter},{stepdata['accuracy']}\n")
        return stepdata

    if write_results_to_file:
        with open(results_file, "w") as file:
            file.write("step,accuracy\n")

    print("Training...")
    num_steps = args.num_epochs * data_size // BATCH_SIZE
    for step_counter in tqdm(range(num_steps), disable=on_cluster):
        key, subkey = random.split(key)
        train_batch = next(train_batches)

        step(subkey, train_batch)
        if step_counter % evaluate_every == 0:
            metrics.append_to_log(particles.rundata,
                                  evaluate(step_counter, particles.particles))
    neural_grad.done()
    particles.done()
    return particles.rundata['step_counter'], particles.rundata['accuracy']


if __name__ == "__main__":
    rngkey = random.PRNGKey(0)
    train(rngkey, write_results_to_file=True)
