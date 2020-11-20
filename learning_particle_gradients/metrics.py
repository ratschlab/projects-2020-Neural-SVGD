from functools import partial

import jax.numpy as np
from jax import grad, vmap, random, jacfwd, jacrev
from jax.scipy import stats, special
from jax.ops import index_update, index
from scipy.spatial.distance import cdist
import ot
import numpy as onp

import utils
import stein
import kernels

def append_to_log(dct, update_dict):
    """appends update_dict to dict, entry-wise. Creates list entry
    if it doesn't exist.
    """
    for key, newvalue in update_dict.items():
        dct.setdefault(key, []).append(newvalue)
    return dct

###############
# Metrics
from scipy.spatial.distance import cdist
import ot
def compute_final_metrics(particles, target):
    """
    Compute the ARD KSD between particles and target.
    particles: np.array of shape (n, d)
    """
    n = len(particles)
    particles = np.asarray(particles)

    target_sample = target.sample(n)
    emd = wasserstein_distance(particles, target_sample)
#    sinkhorn_divergence = ot.bregman.empirical_sinkhorn_divergence(particles, target_sample, 1, metric="sqeuclidean")
#    sinkhorn_divergence = onp.squeeze(sinkhorn_divergence)
    ksd = stein.ksd_squared_u(particles, target.logpdf, kernels.get_rbf_kernel_logscaled(0), False)
    se_mean = np.mean((np.mean(particles, axis=0) - target.mean)**2)
    se_std = np.mean((np.std(particles, axis=0) - np.sqrt(np.diag(target.cov)))**2)
    return dict(emd=emd, ksd=ksd, se_mean=se_mean, se_std=se_std)

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

## KL Divergence
def estimate_kl(logq: callable, logp: callable, samples):
    return np.mean(vmap(logq)(samples) - vmap(logp)(samples))

def _pushforward_log(logpdf: callable, tinv: callable):
    """
    Arguments
        logpdf computes log(p(_)), where p is a PDF.
        tinv is the inverse of an injective transformation T: R^d --> R^d, x --> z

    Returns
        $\log p_T(z)$, where z = T(x). That is, the pushforward log pdf
        $$\log p_T(z) = \log p(T^{-1} z) + \log \det(J_{T^{-1} z})$$
    """
    def pushforward_logpdf(z):
        det = np.linalg.det(jacfwd(tinv)(z))
#         if np.abs(det) < 0.001:
#             raise LinalgError("Determinant too small: T is not injective.")
        return logpdf(tinv(z)) + np.log(np.abs(det))
    return pushforward_logpdf

def pushforward_loglikelihood(t: callable, loglikelihood, samples):
    """
    Compute log p_T(T(x)) for all x in samples.

    Arguments
        t: an injective transformation T: R^d --> R^d, x --> z
        loglikelihood: np array of shape (n, d)
        samples: samples from p, shape (n, d)

    Returns
        np array of shape (n,): $\log p_T(z)$, where z = T(x) for all x in samples.
    That is, the pushforward log pdf
        $$\log p_T(z) = \log p(x) - \log \det(J_T x)$$
    """
    return loglikelihood - np.log(compute_jacdet(t, samples))

def compute_jacdet(t, samples):
    """Just computes the determinants of jacobians.
    Returns np array of shape (n,)"""
    def jacdet(x):
        return np.abs(np.linalg.det(jacfwd(t)(x)))
    return vmap(jacdet)(samples)

def kl_diff(logq, logp, x, transform):
    """
    Arguments:
        logq: computes log(q(x))
        logp: computes log(p(x))
        x: n samples from q, shape (n, d)
        transform: function T: R^d --> R^d, x --> z
    """
    # KL(q || p)
    kl1 = estimate_kl(logq, logp, x)

    # KL(q_T || p) = KL(q || p_{T^{-1}})
    z = vmap(transform)(x)
    logp_pullback = pushforward_log(logp, transform)
    kl2 = estimate_kl(logq, logp_pullback, x)
    return kl1 - kl2

def get_mmd(kernel=kernels.get_rbf_kernel(1.)):
    """
    kernel(x, y) outputs scalar
    """
    kernel_matrix = vmap(vmap(kernel, (0, None)), (None, 0))

    def mmd(xs, ys):
        """Returns approximation of
        E[k(x, x') + k(y, y') - 2k(x, y)]"""
        kxx = utils.remove_diagonal(kernel_matrix(xs, xs))
        kyy = utils.remove_diagonal(kernel_matrix(ys, ys))
        kxy = kernel_matrix(xs, ys)
        return np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)
    return mmd

def get_mmd_tracer(target_samples, kernel=kernels.get_rbf_kernel(1.)):
    mmd = get_mmd(kernel)
    def compute_mmd(particles):
        return {"mmd": mmd(particles, target_samples)}
    return compute_mmd


def get_funnel_tracer(target_samples):
    rbf_mmd = get_mmd(kernels.get_rbf_kernel(1.))
    funnel_mmd = get_mmd(kernels.get_funnel_kernel(1.))
    def compute_mmd(particles):
        return {"rbf_mmd": rbf_mmd(particles, target_samples),
                "funnel_mmd": funnel_mmd(particles, target_samples)}
    return compute_mmd

def get_squared_error_tracer(target_statistic: np.ndarray, statistic: callable, name: str):
    """statistic is a callable that takes in particles and returns
    a value. If the value is not scalar, then the final squared
    error is summed across components. value must be of same shape
    as target_statistic."""
    def compute_summed_error(particles: np.ndarray):
        """compute squared error for each component, then
        sum across components."""
        return {name: np.sum((statistic(particles) - target_statistic)**2)}
    return compute_summed_error

def get_2nd_moment_tracer(target_2nd_moment):
    return get_squared_error_tracer(
        target_2nd_moment,
        lambda particles: np.mean(particles**2, axis=0),
        "second_error"
    )

