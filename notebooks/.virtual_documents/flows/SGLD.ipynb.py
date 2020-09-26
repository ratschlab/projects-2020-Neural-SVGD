get_ipython().run_line_magic("load_ext", " autoreload")
from jax import config
config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")
import json_tricks as json
import copy
from functools import partial

from tqdm import tqdm
import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import matplotlib.pyplot as plt
import numpy as onp
import jax
import pandas as pd
import haiku as hk
from jax.experimental import optimizers

import config

import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import models
import flows

from jax.experimental import optimizers

key = random.PRNGKey(0)


get_ipython().run_line_magic("matplotlib", " inline")
# setup = distributions.banana_target
# target, proposal = setup.get()
target = distributions.Banana([0, 0], [4, 1])
proposal = distributions.Gaussian([-5, -5], 1)
setup = distributions.Setup(target, proposal)
setup.plot(lims=(-15, 15))


n_particles = 900
particles = models.Particles(key=key, gradient=None, proposal=proposal, n_particles=n_particles)


n_steps = 1000
noise = 1.
particle_lr = 0.5
learner_lr = 1e-3

key, subkey = random.split(key)







