# Train a Bayesian neural network to classify MNIST using
# (parallel) Langevin dynamics
#
# If using pmap, set the environment variable
# `export XLA_FLAGS="--xla_force_host_platform_device_count=8"`
# before running on CPU (this enables pmap to "see" multiple cores).
import os
import argparse
from itertools import cycle

import numpy as onp
from jax import jit, value_and_grad, vmap, pmap, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import optax
import utils
from convnet import model, crossentropy_loss, log_prior, ensemble_accuracy
import config as cfg
import jax.flatten_util

on_cluster = not os.getenv("HOME") == "/home/lauro"

# cli args
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=100, help="Number of parallel chains")
parser.add_argument("--num_epochs", type=int, default=1)
args = parser.parse_args()

# Config
key = random.PRNGKey(0)
BATCH_SIZE = 128
LEARNING_RATE = 1e-7
DISABLE_PROGRESS_BAR = on_cluster
USE_PMAP = False
NUM_EVALS = 30  # nr accuracy evaluations

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


def make_batches(images, labels, batch_size):
    """Returns an iterator that cycles through
    tuples (image_batch, label_batch)."""
    num_batches = len(images) // batch_size
    split_idx = onp.arange(1, num_batches+1)*batch_size
    batches = zip(*[onp.split(data, split_idx, axis=0) for data in (images, labels)])
    return cycle(batches)


def loss(params, images, labels):
    """Minibatch approximation of the (unnormalized) Bayesian
    negative log-posterior evaluated at `params`. That is,
    -log model_likelihood(data_batch | params) * batch_rescaling_constant - log prior(params))"""
    logits = model.apply(params, images)
    return data_size/BATCH_SIZE * crossentropy_loss(logits, labels) - log_prior(params)


opt = utils.sgld(LEARNING_RATE)


@jit
def step(param_set, opt_state, images, labels):
    """Update param_set elements in parallel using Langevin dynamics."""
    step_losses, g = vpmap(value_and_grad(loss), (0, None, None))(param_set, images, labels)
    g, opt_state = opt.update(g, opt_state, param_set)
    return optax.apply_updates(param_set, g), opt_state, step_losses


print("Initializing parameters...")
# initialize set of parameters
key, subkey = random.split(key)
param_set = vmap(model.init, (0, None))(random.split(subkey, args.num_samples), train_images[:5])
opt_state = opt.init(param_set)

# save accuracy to file
results_file = cfg.results_path + "bnn-langevin.csv"
with open(results_file, "w") as file:
    file.write("step,accuracy\n")

print("Training...")
# training loop
losses = []
accuracies = []
batches = make_batches(train_images, train_labels, BATCH_SIZE)
n_train_steps = args.num_epochs * data_size // BATCH_SIZE
for step_counter in tqdm(range(n_train_steps), disable=DISABLE_PROGRESS_BAR):
    images, labels = next(batches)
    param_set, opt_state, step_losses = step(param_set, opt_state, images, labels)
    losses.append(step_losses)

    if step_counter % (n_train_steps // NUM_EVALS) == 0:
        acc = ensemble_accuracy(param_set, val_images[:BATCH_SIZE], val_labels[:BATCH_SIZE])
        accuracies.append(acc)
        print(f"Step {step_counter}, Accuracy:", acc)
        print(f"Particle mean: {jax.flatten_util.ravel_pytree(param_set)[0].mean()}")
        with open(results_file, "a") as file:
            file.write(f"{step_counter},{acc}\n")

final_acc = ensemble_accuracy(param_set, val_images[:BATCH_SIZE], val_labels[:BATCH_SIZE])
print(f"Final accuracy: {final_acc}")
with open(results_file, "a") as file:
    file.write(f"{step_counter},{final_acc}\n")
