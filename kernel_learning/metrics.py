from functools import partial

import jax.numpy as np
from jax import grad, vmap, random, jacfwd, jacrev
from jax.scipy import stats, special
from jax.ops import index_update, index
from scipy.spatial.distance import cdist
import ot
import numpy as onp

import svgd
import utils
import stein
import kernels

def append_to_log(dct, update_dict):
    """appends update_dict to dict, entry-wise.
    """
    for key, newvalue in update_dict.items():
        dct.setdefault(key, []).append(newvalue)
    return dct

###############
# Metrics
from scipy.spatial.distance import cdist
import ot
def compute_final_metrics(particles, svgd):
    """
    Compute the ARD KSD between particles and target.
    particles: np.array of shape (n, d)
    svgd: instance of svgd.SVGD
    """
    n = len(particles)

    target_sample = svgd.target.sample(n)
    emd = wasserstein_distance(particles, target_sample)
#    sinkhorn_divergence = ot.bregman.empirical_sinkhorn_divergence(particles, target_sample, 1, metric="sqeuclidean")
#    sinkhorn_divergence = onp.squeeze(sinkhorn_divergence)
    ksd = stein.ksd_squared_u(particles, svgd.target.logpdf, kernels.get_rbf_kernel_logscaled(0), False)
    se_mean = np.mean((np.mean(particles, axis=0) - svgd.target.mean)**2)
    se_var = np.mean((np.cov(particles, rowvar=False) - svgd.target.cov)**2)
    return dict(emd=emd, ksd=ksd, se_mean=se_mean, se_var=se_var)

def wasserstein_distance(s1, s2):
    """
    Arguments: samples from two distributions, shape (n, d) (not (n,)).
    Returns: W2 distance inf E[d(X, Y)^2]^0.5 over all joint distributions of X and Y such that the marginal distributions are equal those of the input samples. Here, d is the euclidean distance."""
    M = cdist(s1, s2, "sqeuclidean")
    a = np.ones(len(s1)) / len(s1)
    b = np.ones(len(s2)) / len(s2)
    return np.sqrt(ot.emd2(a, b, M))

def sqrt_kxx(kernel: callable, particles_a, particles_b):
    """Approximate E[k(x, x)] in O(n^2)"""
    def sqrt_k(x, y): return np.sqrt(kernel(x, y))
    sv  = vmap(sqrt_k, (0, None))
    svv = vmap(sv,     (None, 0))
    return np.mean(svv(particles_a, particles_b))
#    return np.mean(vmap(sqrt_k)(particles_a, particles_b))

### Some nice target distributions
# 2D
l = np.asarray((1, 2, 1.5, 3, 3.3, 3.8))
l = onp.concatenate([-l, [0], l])
means = list(zip(l, (l**2)**0.8))
variances = [[1,1]]*len(means)
weights = [1]*len(means)
#bent = GaussianMixture(means, variances, weights)
bent_args = [means, variances, weights]
