# get_ipython().run_line_magic("load_ext", " autoreload")
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

from jax.experimental import optimizers

key = random.PRNGKey(43)

import matplotlib.pyplot as plt
from scipy.stats import kde


import time
import pylab as pl
from IPython import display
for i in range(10):
    pl.clf()
    pl.plot(pl.randn(100))
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(1.0)


from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
from threading import Thread
import time

class LiveGraph:
    def __init__(self):
        self.x_data, self.y_data = [], []
        self.figure = plt.figure()
        self.line, = plt.plot(self.x_data, self.y_data)
        self.animation = FuncAnimation(self.figure, self.update, interval=1000)
        self.th = Thread(target=self.thread_f, daemon=True)
        self.th.start()

    def update(self, frame):
        self.line.set_data(self.x_data, self.y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()
        return self.line,

    def show(self):
        plt.show()

    def thread_f(self):
        x = 0
        while True:
            self.x_data.append(x)
            x += 1
            self.y_data.append(randrange(0, 100))   
            time.sleep(1)  

g = LiveGraph()
g.show()


sdkfj


density = distributions.banana


plt.pcolormesh(*plot.make_meshgrid(density.pdf, lims=(-10,10)))


plt.pcolormesh(*plot.make_meshgrid(density.logpdf, lims=(-10,10)), vmin=-15, vmax=0)


grad_density1 = grad(density.pdf)
def grad_density(x, y):
    return grad_density1(np.append(x, y))


def quiverplot(f, num_gridpoints=50, lims=[-10, 10], xlims=None, ylims=None, angles="xy", scale=2, **kwargs):
    """
    Plot a vector field. f is a function f: R^2 ---> R^2
    If arrows are too large, change scale (larger scale = shorter arrows)
    """
    if xlims is None:
        xlims = lims
    if ylims is None:
        ylims = lims

    def split_f(x, y):
        return f(np.append(x, y))

    x = np.linspace(*xlims, num_gridpoints)
    y = np.linspace(*ylims, num_gridpoints)
    xx, yy = np.meshgrid(x, y, dtype=np.float32)
    zz = vmap(vmap(split_f))(xx, yy)
    uu, vv = np.rollaxis(zz, 2)

    plt.quiver(xx, yy, uu, vv, angles=angles, scale=scale, **kwargs)


plt.subplots(figsize=[10,8])
plt.pcolormesh(*plot.make_meshgrid(density.pdf, lims=(-10,10)))
quiverplot(grad_density1)
plt.colorbar()
