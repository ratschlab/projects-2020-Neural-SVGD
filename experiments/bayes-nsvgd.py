import os
import argparse
from itertools import cycle
import numpy as onp
import jax
from jax import numpy as jnp
from jax import vmap, pmap, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import optax
from convnet import model, crossentropy_loss, log_prior, ensemble_accuracy
import models
import config as cfg

on_cluster = not os.getenv("HOME") == "/home/lauro"

# cli args
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=20, help="Number of parallel chains")
parser.add_argument("--num_epochs", type=int, default=1)
args = parser.parse_args()

# Config
key = random.PRNGKey(0)
results_file = cfg.results_path + "bnn-nsvgd.csv"
BATCH_SIZE = 128
META_LEARNING_RATE = 1e-3
DISABLE_PROGRESS_BAR = on_cluster
USE_PMAP = False
NUM_EVALS = 30  # nr accuracy evaluations
LAMBDA_REG = 10**2
STEP_SIZE = 1e-7 * LAMBDA_REG * 2 * 50

LAYER_SIZE = 128 if on_cluster else 32
NUM_WARMUP_STEPS = 300 if on_cluster else 100
PATIENCE = 5

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


# utility functions for dealing with parameters
key, subkey = random.split(key)
params_tree = model.init(subkey, train_images[:2])
params_flat, unravel = jax.flatten_util.ravel_pytree(params_tree)


def ravel(tree):
    return jax.flatten_util.ravel_pytree(tree)[0]


def init_flat_params(key):
    return ravel(model.init(key, train_images[:2]))


def get_minibatch_loss(batch):
    """
    Returns a callable that computes target posterior
    given flattened param vector.

    args:
        batch: tuple (images, labels)
    """
    def minibatch_loss(params_flat):
        return loss(unravel(params_flat), *batch)
    return minibatch_loss


def sample_tv(key):
    """return two sets of particles at initialization, for
    training and validation in the warmup phase"""
    return vmap(init_flat_params)(random.split(subkey, args.num_samples)).split(2)


def compute_acc(param_set_flat):
    return ensemble_accuracy(vmap(unravel)(param_set_flat),
                             val_images[:BATCH_SIZE],
                             val_labels[:BATCH_SIZE])


def vmean(fun):
    """vmap, but computes mean along mapped axis"""
    def compute_mean(*args, **kwargs):
        return jnp.mean(vmap(fun)(*args, **kwargs), axis=-1)
    return compute_mean


# initialize particles and the dynamics model
key, subkey = random.split(key)
init_particles = vmap(init_flat_params)(random.split(subkey, args.num_samples))

opt = optax.sgd(STEP_SIZE)

key, subkey1, subkey2 = random.split(key, 3)
neural_grad = models.SDLearner(target_dim=init_particles.shape[1],
                               get_target_logp=get_minibatch_loss,
                               learning_rate=META_LEARNING_RATE,
                               key=subkey1,
                               sizes=[LAYER_SIZE, LAYER_SIZE, init_particles.shape[1]],
                               aux=False,
                               use_hutchinson=True,
                               lambda_reg=LAMBDA_REG,
                               patience=PATIENCE)
particles = models.Particles(subkey2, neural_grad.gradient, init_particles, custom_optimizer=opt)


# Training
###########
train_batches = make_batches(train_images, train_labels, BATCH_SIZE)
test_batches  = make_batches(test_images, test_labels, BATCH_SIZE)

# Warmup on first batch
print("Warmup...")
neural_grad.train(next_batch=sample_tv,
                  n_steps=NUM_WARMUP_STEPS,  # 100
                  early_stopping=False,
                  data=next(train_batches),
                  progress_bar=not on_cluster)


# training loop
def step(train_batch):
    """one iteration of the particle trajectory simulation"""
    neural_grad.train(next_batch=particles.next_batch, n_steps=50, data=train_batch)
    particles.step(neural_grad.get_params())
    return


def evaluate(step_counter):
    acc = compute_acc(particles.particles).tolist()
    print(f"Step {step_counter}, accuracy: {acc}")
    print("particle mean: ", onp.mean(particles.particles))
    with open(results_file, "a") as file:
        file.write(f"{step_counter},{acc}\n")
    return


with open(results_file, "w") as file:
    file.write("step,accuracy\n")

print("Training...")
num_steps = args.num_epochs * data_size // BATCH_SIZE
for step_counter in tqdm(range(num_steps), disable=on_cluster):
    train_batch = next(train_batches)
    step(train_batch)
    if step_counter % (num_steps//NUM_EVALS) == 0:
        evaluate(step_counter)

neural_grad.done()
particles.done()
