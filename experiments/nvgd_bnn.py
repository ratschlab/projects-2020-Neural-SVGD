import os
import argparse
from jax import vmap, random, jit, grad
from tqdm import tqdm
import optax
import bnn
import models
import metrics
import mnist
import config as cfg
import pandas as pd

on_cluster = not os.getenv("HOME") == "/home/lauro"

# Config
# date = datetime.today().strftime('%a-%H:%M-%f')
DEFAULT_MAX_TRAIN_STEPS = 100
DEFAULT_META_LR = 1e-3  # should be as high as possible; regularize w/ max steps
DEFAULT_PATIENCE = 5  # early stopping not v helpful, bc we overfit on all ps

LAMBDA_REG = 100
LAYER_SIZE = 256 if on_cluster else 32


def train(key,
          meta_lr: float = DEFAULT_META_LR,
          particle_stepsize: float = 1e-3,
          evaluate_every: int = 10,
          n_iter: int = 200,
          n_samples: int = cfg.n_samples+1, # add 1 to account for dummy val set
          particle_steps_per_iter: int = 1,
          max_train_steps_per_iter: int = DEFAULT_MAX_TRAIN_STEPS,
          patience: int = DEFAULT_PATIENCE,
          dropout: bool = True,
          results_file: str = cfg.results_path + 'nvgd-bnn.csv',
          overwrite_file: bool = False,
          early_stopping: bool = True):
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
    init_particles = vmap(bnn.init_flat_params)(random.split(subkey, n_samples))
    opt = optax.sgd(particle_stepsize)
    print(f"particle shape: {init_particles.shape}")

    key, subkey1, subkey2 = random.split(key, 3)
    neural_grad = models.SDLearner(target_dim=init_particles.shape[1],
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

    minibatch_vdlogp = vmap(value_and_grad(bnn.minibatch_logp), (0, None))

    @jit
    def split_vdlogp(split_particles, train_batch):
        """returns tuple (split_logp, split_dlogp)"""
        train_out, val_out = [minibatch_vdlogp(x, train_batch)
                              for x in split_particles]
        return tuple(zip(train_out, val_out))

    def step(split_particles, split_dlogp):
        """one iteration of the particle trajectory simulation"""
        neural_grad.train(split_particles=split_particles,
                          split_dlogp=split_dlogp,
                          n_steps=max_train_steps_per_iter,
                          early_stopping=early_stopping)
        for _ in range(particle_steps_per_iter):
            particles.step(neural_grad.get_params())
        return

    def evaluate(step_counter, ps, logp):
        ll = logp.mean()
        stepdata = {
            "accuracy": bnn.compute_acc_from_flat(ps),
            "step_counter": step_counter,
            "loglikelihood": ll,
        }
        with open(results_file, "a") as file:
            file.write(csv_string + f"{step_counter},{stepdata['accuracy']},{ll}\n")
        return stepdata

    if not os.path.isfile(results_file) or overwrite_file:
        with open(results_file, "w") as file:
            file.write("step_counter,accuracy,loglikelihood\n")

    print("Training...")
    for step_counter in tqdm(range(n_iter), disable=on_cluster):
        key, subkey = random.split(key)
        train_batch = next(mnist.training_batches)
        n_train_particles = 3*n_samples // 4 if early_stopping else n_samples - 1
        split_particles = particles.next_batch(key, n_train_particles=n_train_particles)
        split_loss, split_dlogp = split_vdlogp(split_particles, train_batch)
        step(split_particles, split_dlogp)

        if (step_counter+1) % evaluate_every == 0:
            eval_ps = particles.particles if early_stopping else split_particles[0]
            metrics.append_to_log(particles.rundata,
                                  evaluate(step_counter, eval_ps))

        if step_counter % mnist.steps_per_epoch == 0:
            print(f"Starting epoch {step_counter // mnist.steps_per_epoch + 1}")

    neural_grad.done()
    particles.done()

    final_eval = evaluate(-1, particles.particles, split_particles[0])
    return final_eval['accuracy'], particles.rundata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of parallel chains")
    parser.add_argument("--n_iter", type=int, default=200)
    args = parser.parse_args()

    print("Loading optimal step size")
    results_file = cfg.results_path + "nvgd-bnn.csv"
    stepsize_csv = cfg.results_path + "bnn-sweep/best-stepsizes.csv"
    try:
        sweep_results = pd.read_csv(stepsize_csv, index_col=0)
        stepsize = sweep_results['optimal_stepsize']['nvgd']
    except (FileNotFoundError, TypeError):
        print('CSV sweep results not found; using default')
        stepsize = 1e-3

    rngkey = random.PRNGKey(0)
    train(key=rngkey,
          n_samples=args.n_samples,
          n_iter=args.n_iter,
          particle_stepsize=stepsize)
