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
from convnet import make_model, crossentropy_loss, log_prior, ensemble_accuracy
import models
import metrics
import config as cfg

on_cluster = not os.getenv("HOME") == "/home/lauro"
model = make_model("small")

# Config
# date = datetime.today().strftime('%a-%H:%M-%f')
DISABLE_PROGRESS_BAR = on_cluster
USE_PMAP = False

LAMBDA_REG = 10**2
LAYER_SIZE = 256 if on_cluster else 32
BATCH_SIZE = 1024 if on_cluster else 128

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
steps_per_epoch = data_size // BATCH_SIZE


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


# 6 val batches
validation_batches = make_batches(val_images, val_labels, 1024, cyclic=False)

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


# def sample_tv(key, num_samples):
#     """return two sets of particles at initialization, for
#     training and validation in the warmup phase"""
#     return vmap(init_flat_params)(random.split(key, num_samples)).split(2)


def minibatch_accuracy(param_set_flat, images, labels):
    param_set = vmap(unravel)(param_set_flat)
    logits = vmap(model.apply, (0, None))(param_set, images)
    return ensemble_accuracy(logits, labels)


@jit
def compute_acc(param_set_flat):
    accs = []
    for batch in validation_batches:
        accs.append(minibatch_accuracy(param_set_flat, *batch))
    return jnp.mean(jnp.array(accs))


def vmean(fun):
    """vmap, but computes mean along mapped axis"""
    def compute_mean(*args, **kwargs):
        return jnp.mean(vmap(fun)(*args, **kwargs), axis=-1)
    return compute_mean


def train(key,
          meta_lr: float = 1e-3,
          particle_stepsize: float = 1e-3,
          evaluate_every: int = 10,
          n_iter: int = 400,
          n_samples: int = 100,
          particle_steps_per_iter: int = 1,
          max_train_steps_per_iter: int = 10,
          patience: int = 0,
          dropout: bool = True,
          results_file: str = cfg.results_path + 'bnn-nsvgd.csv',
          overwrite_file: bool = False):
    """
    Initialize model; warmup; training; evaluation.
    Returns a dictionary of metrics.
    Args:
        meta_lr: learning rate of Stein network
        particle_stepsize: learning rate of BNN
        evaluate_every: compute metrics every `evaluate_every` steps
        n_iter: number of train-update iterations
        particle_steps_per_iter: num particle updates after training Stein network
        max_train_steps_per_iter: cutoff for Stein network training iteration
        patience: early stopping criterion
        dropout: use dropout during training of the Stein network
        write_results_to_file: whether to save accuracy in csv file
    """
    csv_string = f"{meta_lr},{particle_stepsize}," \
                 f"{patience},{max_train_steps_per_iter},"

    # initialize particles and the dynamics model
    key, subkey = random.split(key)
    init_particles = vmap(init_flat_params)(random.split(subkey, n_samples))
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
                          n_steps=max_train_steps_per_iter,
                          data=train_batch)
        for _ in range(particle_steps_per_iter):
            particles.step(neural_grad.get_params())
        return

    def evaluate(step_counter, ps):
        stepdata = {
            "accuracy": compute_acc(ps),
            "step_counter": step_counter,
        }
        with open(results_file, "a") as file:
            file.write(csv_string + f"{step_counter},{stepdata['accuracy']}\n")
        return stepdata

    if not os.path.isfile(results_file) or overwrite_file:
        with open(results_file, "w") as file:
            file.write("meta_lr,particle_stepsize,patience,"
                       "max_train_steps,step,accuracy\n")

    print("Training...")
    for step_counter in tqdm(range(n_iter), disable=on_cluster):
        key, subkey = random.split(key)
        train_batch = next(train_batches)
        step(subkey, train_batch)

        if step_counter % evaluate_every == 0:
            metrics.append_to_log(particles.rundata,
                                  evaluate(step_counter, particles.particles))

        if (step_counter+1) % steps_per_epoch == 0:
            print(f"Starting epoch {step_counter // steps_per_epoch + 1}")

    neural_grad.done()
    particles.done()
    return particles.rundata['step_counter'], particles.rundata['accuracy']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100, help="Number of parallel chains")
    parser.add_argument("--n_epochs", type=int, default=1)
    args = parser.parse_args()

    rngkey = random.PRNGKey(0)
    train(key=rngkey,
          n_samples=args.n_samples,
          n_iter=args.n_epochs * data_size // BATCH_SIZE)
