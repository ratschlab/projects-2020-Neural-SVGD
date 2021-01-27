# Train a Bayesian neural network to classify MNIST using
# (parallel) Langevin dynamics
#
# If using pmap, set the environment variable
# `export XLA_FLAGS="--xla_force_host_platform_device_count=8"`
# before running on CPU (this enables pmap to "see" multiple cores).
import os
import argparse
from jax import jit, value_and_grad, vmap, random
from tqdm import tqdm
import optax
import utils
import config as cfg
import jax.flatten_util
import mnist
import bnn

on_cluster = not os.getenv("HOME") == "/home/lauro"

# Config
key = random.PRNGKey(0)
LEARNING_RATE = 1e-7
DISABLE_PROGRESS_BAR = on_cluster
NUM_EVALS = 30  # nr accuracy evaluations


def train(key,
          particle_stepsize: float = 1e-7,
          n_samples: int = 100,
          n_epochs: int = 1):
    """Train langevin BNN"""
    opt = utils.sgld(particle_stepsize)

    print("Initializing parameters...")
    key, subkey = random.split(key)
    param_set = vmap(bnn.model.init, (0, None))(
        random.split(subkey, n_samples), mnist.train_images[:5])
    opt_state = opt.init(param_set)

    # save accuracy to file
    results_file = cfg.results_path + "bnn-langevin.csv"
    with open(results_file, "w") as file:
        file.write("step,accuracy\n")

    @jit
    def step(param_set, opt_state, images, labels):
        """Update param_set elements in parallel using Langevin dynamics."""
        step_losses, g = vmap(value_and_grad(bnn.loss), (0, None, None))(
            param_set, images, labels)
        g, opt_state = opt.update(g, opt_state, param_set)
        return optax.apply_updates(param_set, g), opt_state, step_losses

    print("Training...")
    # training loop
    losses = []
    accuracies = []
    n_train_steps = n_epochs * mnist.train_data_size // cfg.batch_size
    for step_counter in tqdm(range(n_train_steps), disable=DISABLE_PROGRESS_BAR):
        images, labels = next(mnist.training_batches)
        param_set, opt_state, step_losses = step(param_set, opt_state, images, labels)
        losses.append(step_losses)

        if step_counter % (n_train_steps // NUM_EVALS) == 0:
            acc = bnn.compute_acc(param_set)
            accuracies.append(acc)
            print(f"Step {step_counter}, Accuracy:", acc)
            print(f"Particle mean: {jax.flatten_util.ravel_pytree(param_set)[0].mean()}")
            with open(results_file, "a") as file:
                file.write(f"{step_counter},{acc}\n")

    return bnn.compute_acc(param_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100, help="Number of parallel chains")
    parser.add_argument("--n_epochs", type=int, default=1)
    args = parser.parse_args()

    rngkey = random.PRNGKey(0)
    train(key=rngkey,
          n_samples=args.n_samples,
          n_epochs=args.n_epochs)