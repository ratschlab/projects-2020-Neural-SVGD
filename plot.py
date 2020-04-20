import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

def bivariate_hist(xout):
    def myplot(x, y, s, bins=1000):
        heatmap, xedges, yedges = onp.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 8)
    plt.axis('equal')

    x = xout[:, 0]
    y = xout[:, 1]

    sigmas = [0, 32, 64]

    for ax, s in zip(axs.flatten(), sigmas):
        ax.set_aspect("equal")
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title("Scatter plot")
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title("Smoothing with  $\sigma$ = %d" % s)
    plt.show()

def svgd_log(log, xout=None):
    """plot metrics logged during SVGD run and histogram of output."""
    plt.figure(1, figsize = [20, 10])
    plt.subplots_adjust(hspace=0.8)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    l = len(log)
    k = l // 2
    print(l, k)
    for i, key in enumerate(log.keys()):
        plt.subplot(f"{k}{l-k}{i+1}") # 2 plots on 0th axis, 1 plot on 1th axis, plot nr 1 --> 211
        plt.plot(log[key], color=colors[i])
        plt.title(key)
        plt.xlabel("step")
        if "ksd" in key:
            plt.yscale("log")

    if xout is not None:
        plt.figure(2)
        _ = plt.hist(xout[:, 0], density=True, bins=25)
