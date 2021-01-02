# Overview

Many standard MCMC methods such as Hamiltonian Monte Carlo don't work well in settings with large data-sets and high-dimensional target posteriors with complicated dependencies. This is why usually simpler methods such as variational inference (VI) or stochastic gradient Langevin dynamics (SGLD) are applied to this type of problem (e.g. training a Bayesian neural network).

In 2016, Quang Liu and Dilin Wang proposed Stein variational gradient descent (SVGD), a new kind of inference method that rapidly became popular. SVGD transports a set of particles $x_1, \dots, x_n$ along a trajectory that (approximately) minimizes the KL divergence to the target. In contrast to most Markov chain Monte Carlo (MCMC) methods, it does so by leveraging interactions between the $n$ particles. Here's an animation of the process (the blue samples represent the target density):

![](/home/lauro/code/msc-thesis/main/illustrations/svgd.gif)

A drawback of SVGD is that it is dependent on the choice of a kernel function. If this kernel is not chosen well, the method may converge badly or not at all. This is reflected in the observation that SVGD is not robust in high dimensions (cite). 

The goal of this project is to build an alternative to SVGD that does not depend on a choice of kernel.

# Organization

All code is contained in the `learning_particle_gradients` folder. The files are structured as follows:

* `distributions.py`: a set of classes that bundle together all attributes and methods associated with a probability distribution (e.g. mean, variance, sampling, computing the likelihood and loglikelihood).
* `flows.py`: implements functions to simulate the particle dynamics, using the models in `models.py`
* `kernels.py`: a set of positive definite kernel functions.
* `metrics.py`: utilities for computing metrics to track convergence to the target (e.g. MMD distance or mean squared error).
* `models.py`: this is the heart of the project. Contains different models that each compute a single iteration of their respective particle dynamics.
* `nets.py`: neural network architectures.
* `plot.py`: utility functions for plotting.
* `stein.py`: implementations of the Stein operator, (kernelized) Stein discrepancy, and associated methods.
* `utils.py`: miscellaneous utility functions.

# Dependencies


