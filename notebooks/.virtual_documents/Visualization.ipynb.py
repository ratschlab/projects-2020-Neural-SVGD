import matplotlib.pyplot as plt
import jax.numpy as np
from jax import grad, vmap, jacfwd
from scipy.stats import kde
import sys
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning/")

import distributions
import plot


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
