import numpy as onp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
# from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as np
from jax import vmap

import metrics

## plotting utilities
def equalize_axes(ax):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    lim = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    return lim

def bivariate_hist(xout):
    def myplot(x, y, s, bins=1000):
        heatmap, xlim, ylim = onp.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)

#        lim = [min(ylim[0], xlim[0]), max(ylim[-1], xlim[-1])]
#        extent = lim + lim
        extent = [xlim[0], xlim[-1], ylim[0], ylim[-1]]
        return heatmap.T, extent

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 8)
#    plt.axis('equal')

    x = xout[:, 0]
    y = xout[:, 1]

    sigmas = [0, 32, 64]

    for ax, s in zip(axs.flatten(), sigmas):
        ax.set_aspect("equal")
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title("Scatter plot")
            equalize_axes(ax)
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title("Smoothing with  $\sigma$ = %d" % s)
        equalize_axes(ax)
#        ax.set_aspect("equal")
    plt.show()


def plotobject(data, colors=None, titles=None, xscale="linear", yscale="linear", xlabel=None, ylabel=None, style="-", xaxis=None, axhlines=None):
    """
    * if data is a dict, plot every value.
    * if data is an array, iterate over first axis and plot
    """
    assert type(data) is dict or data.ndim <= 3
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sq = int(np.sqrt(len(data)))
    w = sq + 3
    h = sq
    plt.figure(figsize = [6*w, 2.5*h + 0.2*(h-1)]) # 0.2 = hspace
    if type(data) is dict:
        for i, (k, v) in enumerate(sorted(data.items())):
            plt.subplot(f"{h}{w}{i+1}")
            plt.plot(v, style, color=colors[i])
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.title(k)
    else:
        for i, v in enumerate(data):
            plt.subplot(f"{h}{w}{i+1}")
            if xaxis is None:
                plt.plot(v, style, color=colors[i])
            else:
                plt.plot(xaxis, v, style, color=colors[i])
            plt.title(titles[i])
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if axhlines is not None:
                plt.axhline(y=axhlines[i], color="y")


def svgd_log(log, style="-", full=False):
    """plot metrics logged during SVGD run."""
    # plot mean and var
    titles = metrics.Distribution.metric_names
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors + colors + colors # avoid index out of bound
    for key, dic in log.items():
        if key == "desc":
            plotobject(dic, colors, style=style)
            colors = colors[len(dic):]

        elif key == "metrics" and full:
            for k, v in dic.items():
                v = np.moveaxis(v, 0, 1)
                plotobject(v, colors, titles[k], yscale="log", style=style) # moveaxis swaps axes 0 and 1
                colors = colors[len(v):]


def make_meshgrid(func, lims, num=100):
    """
    Arguments:
    * func: callable. Takes an np.array of shape (d,) as only input, and outputs a scalar.
    """
    x = y = np.linspace(*lims, num)
    xx, yy = np.meshgrid(x, y)

    grid = np.stack([xx, yy]) # shape (2, r, r)
    zz = vmap(vmap(func, 1), 2)(grid)

    #zz = np.exp(zz)
    return xx, yy, zz


def plot_3d(x, y, z):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z,
                  cmap=cm.coolwarm,
                  linewidth=0,
                  antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()
    if ax is None: print("huh?")
    return ax

def plot_pdf(pdf, lims):
    return plot_3d(*make_meshgrid(pdf, lims, num=150))
