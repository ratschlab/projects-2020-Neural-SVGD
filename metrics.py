import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd, jacrev
from jax.ops import index_update, index
from jax.scipy import stats

import numpy as onp

import svgd
import utils

def log_gaussian_mixture(x, means, variances, weights):
    """
    IN:
    * x: scalar or array of length n
    * mean: np array of means
    * variances: np array
    * weights: np array of weights, same length as the mean and varance arrays

    OUT:
    scalar, value of log(p(x)), where p is the mixture pdf.
    """

# distributions packaged with metrics
# check wikipedia for computation of higher moments https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Higher_moments
# also recall form of characteristic function https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)#Examples
class Distribution():
    """Base class for package logpdf + metrics + sampling"""
    def __init__(self):
        pass

    def newkey(self):
        self.key = random.split(self.key)[0]

    def sample(self, shape):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def compute_metrics(self, x):
        """Compute metrics given samples"""
        sample_expectations = [np.mean(value, axis=0) for value in (x, x**2, np.cos(x), np.sin(x))]
        square_errors = [(sample - true)**2 for sample, true in zip(sample_expectations, self.expectations)]
        square_errors = np.array(square_errors) # shape (4, d)

        ksds = [ksd(x, self.logpdf, h) for h in (0.1, 1, 10)]

        metrics_dict = {
            "square_errors": square_errors,
            "ksds": ksds
        }
        return metrics_dict

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        """self.expectations is a list of expected values of the following expressions: x, x^2, cos(x), sin(x)"""
        mean = np.array(mean)
        cov = np.array(cov)
        if cov.ndim == 1:
            cov = np.diag(cov)
        if mean.ndim == 0:
            mean = np.ones(len(cov)) * mean
        assert mean.ndim == 1 and cov.ndim == 2

        self.d = mean.shape[0]
        self.mean = mean
        self.cov = cov
        if not utils.is_pd(cov):
            raise ValueError("Covariance must be positive definite.")

        # characteristic function at [1, ..., 1]:
        t = np.ones(self.d)
        char = np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2)))
        self.expectations = [mean, np.diagonal(cov) + mean**2, np.real(char), np.imag(char)]
        self.key = random.PRNGKey(0)
        return None

    def sample(self, shape):
        self.newkey()
        return random.multivariate_normal(self.key, self.mean, self.cov, shape)

    def logpdf(self, x):
        x = np.array(x)
        assert x.ndim == 1
        return stats.multivariate_normal.logpdf(x, self.mean, self.cov)

class GaussianMixture(Distribution):
    """TODO fix computation of self.cov"""
    def __init__(self, means, covs, weights):
        """means, covs are np arrays or lists of length k, with entries of shape (d,) and (d, d) respectively.
        self.expectations is a list of expected values of the following expressions: x, x^2, cos(x), sin(x)"""
        means = np.array(means)
        covs = np.array(covs)
        weights = np.array(weights)
        weights = weights / np.sum(weights) # normalize

        if means.ndim == 1:
            means = means[:, np.newaxis]
        d = means.shape[1]
        if covs.ndim == 1 or covs.ndim == 2:
            covs = [np.identity(d) * var for var in covs]
            covs = np.array(covs)

        assert weights.ndim == 1 and len(weights) > 1
        assert means.ndim == 2 and covs.ndim == 3
        assert covs.shape[1] == covs.shape[2]

        for cov in covs:
            if not utils.is_pd(cov):
                raise ValueError("Covariance must be positive definite.")

        # characteristic function at [1, ..., 1]:
        t = np.ones(d)
        chars = np.array([np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2))) for mean, cov in zip(means, covs)]) # shape (k,d)
        char = np.vdot(weights, chars)

        mean = np.einsum("i,id->d", weights, means)
        xsquares = [np.diagonal(cov) + mean**2 for mean, cov in zip(means, covs)]
        self.expectations = [mean, np.einsum("i,id->d", weights, xsquares), np.real(char), np.imag(char)]
        self.key = random.PRNGKey(0)
        self.means = means
        self.covs = covs
        self.weights = weights
        self.num_components = len(weights)
        self.d = d # particle dimension
        self.mean = mean
        # recall Cov(X) = E[XX^T] - mu mu^T = sum_over_components(Cov(Xi) + mui mui^T) - mu mu^T
        mumut = np.einsum("ki,kj->kij", means, means) # shape (k, d, d)
        self.cov = np.average(covs + mumut, weights=weights, axis=0)

    def sample(self, shape):
        """mutates self.rkey"""
        def csample(rkey, component, num_samples):
            return random.multivariate_normal(rkey, self.means[component], self.covs[component], shape=(num_samples,))
        components = random.categorical(self.key, np.log(self.weights), shape=shape)
        counts = onp.bincount(components.flatten())
        self.newkey()

        out = [csample(key, c, num) for key, c, num in zip(random.split(self.key, self.num_components), range(self.num_components), counts)]
        out = np.concatenate(out)
        self.newkey()
        return out.reshape(shape + (self.d,))

    def logpdf(self, x):
        """unnormalized"""
        x = np.array(x)

        def exponent(x, mean, cov):
            sigmax = np.dot(np.linalg.inv(cov), (x - mean))
            return - np.vdot((x - mean), sigmax) / 2
        exponents = vmap(exponent, (None, 0, 0))(x, self.means, self.covs) + np.log(self.weights)

        out = logsumexp(exponents)
        return np.squeeze(out)


######################################
# Kernelized Stein Discrepancy
def ksd(x, logp, bandwidth):
    """
    Arguments:
    * x: np.array of shape (n, d)
    * logp: callable, takes in single input x of shape (d,)
    * bandwidth: scalar or np.array of shape (d,)
    Returns:
    * scalar representing the estimated kernelized Stein discrepancy between target p and samples x.
    """
    def ksd_i(xi, x, logp, bandwidth):
        dphi_dxi_transposed = lambda xi: np.trace(jacfwd(svgd.phistar_i)(xi, x, logp, bandwidth))
        return np.vdot(grad(logp)(xi), svgd.phistar_i(xi, x, logp, bandwidth)) + dphi_dxi_transposed(xi)

    return np.mean(vmap(ksd_i, (0, None, None, None))(x, x, logp, bandwidth))
ksd = jit(ksd, static_argnums=1)

def mses(xout):
    """Meant for the case where the target p is the usual univariate gaussian mixture."""
    assert xout.shape[1] == 1
    mse1 = (np.mean(xout) - 2/3)**2
    mse2 = (np.mean(xout**2) - 5)**2
    return mse1, mse2

def get_metrics(xout, logp, ksd_bandwidths):
    """
    Compute metrics. One-dim case. Target p is gaussian mixture.
    """
    assert xout.ndim == 2
    assert xout.shape[1] == 1
    assert not np.any(np.isnan(xout))

    metrics = []
    metrics.extend(mses(xout))
    for ksd_bandwidth in ksd_bandwidths:
        metrics.append(ksd(xout, logp, ksd_bandwidth))
    return metrics


########################
### metrics to log while running SVGD
def initialize_log(self):
    d = self.particle_shape[1]
    log = {
        "particle_mean": np.zeros(shape=(self.n_iter_max, d)),
        "particle_var":  np.zeros(shape=(self.n_iter_max, d)),
        "ksd": np.zeros(shape=(self.n_iter_max, 1))
    }

    for h in self.ksd_kernel_range:
        log[f"ksd {h}"] = np.zeros(shape=(self.n_iter_max, 1))

    if self.adaptive_kernel:
        log["bandwidth"] = np.zeros(shape=(self.n_iter_max, d))
    return log


def update_log(self, i, x, log, bandwidth, adaptive_bandwidth):
    """
    1) create dict of updates
    2) 'append' dict of updates to log dict
    """
    update_dict = {
        "particle_mean": np.mean(x, axis=0),
        "particle_var": np.var(x, axis=0)
    }
    if self.adaptive_kernel:
        update_dict["bandwidth"] = adaptive_bandwidth

    update_dict["ksd"] = ksd(x, self.logp, adaptive_bandwidth if self.adaptive_kernel else bandwidth)
    for h in self.ksd_kernel_range:
        update_dict[f"ksd {h}"] = ksd(x, self.logp, h)

    for key in log.keys():
        log[key] = index_update(log[key], index[i, :], update_dict[key])
    return log
