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

# distributions packaged with metrics and sampling
# check wikipedia for computation of higher moments
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Higher_moments
# also recall form of characteristic function
# https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)#Examples
class Distribution():
    """Base class for package logpdf + metrics + sampling"""
    def __init__(self):
        self.expectations = None
        self.sample_metrics = None
        self.d = None
        self.initialize_metric_names()
        pass

    threadkey = random.PRNGKey(0)

    def newkey(self):
        self.threadkey, self.key = random.split(self.threadkey)

    def sample(self, shape):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def compute_metrics(self, x, normalize=False):
        """Compute metrics given samples x.
        If normalize = True, then all values are divided by the corresponding expected value for a true random sample of the same size."""
        if x.shape[-1] != self.d:
            raise ValueError(f"Particles x need to have shape (n, d), where d = {self.d} is the particle dimension.")
        if normalize:
            n = x.shape[0]
            metrics = self.compute_metrics(x)
            sample_metrics = self.compute_metrics_for_sample(n)
            return utils.dict_divide(metrics, sample_metrics)
        else:
            sample_expectations = [np.mean(value, axis=0) for value in (x, x**2, np.cos(x), np.sin(x))]
            square_errors = [(sample - true)**2 for sample, true in zip(sample_expectations, self.expectations)]
            square_errors = np.array(square_errors)  # shape (4, d)

            metrics_dict = {
                "square_errors": square_errors  # shape (4, d)
            }

            return metrics_dict

    def _checkx(self, x):
        """check if particle (single particle shape (d,)) in right shape, etc"""
        x = np.array(x)
        if x.ndim == 0 and self.d == 1:
            x = x.reshape((1,))
        if x.shape != (self.d,):
            raise ValueError(f"x needs to have shape ({self.d},). Instead, received x of shape {x.shape}.")
        return x

    def get_metrics_shape(self):
        shapes = {
            "square_errors": (4, self.d)
        }
        if self.d == 1:
            shapes["KL Divergence"] = (1,)
        return shapes

    def initialize_metric_names(self):
        self.metric_names = {
            "square_errors": [f"SE for {val}" for val in ["X", "X^2", "cos(X)", "sin(X)"]]
        }
        if self.d == 1:
            self.metric_names["KL Divergence"] = "Estimated KL Divergence"
        return None

    def compute_metrics_for_sample(self, sample_size):
        """For benchmarking. Returns metrics computed for a true random sample of size sample_size, averaged over 100 random seeds."""
        if sample_size not in self.sample_metrics:
            def compute():
                sample = self.sample(shape=(sample_size,))
                sample = np.reshape(sample, newshape=(sample_size, self.d))
                return self.compute_metrics(sample)
            self.sample_metrics[sample_size] = utils.dict_mean([compute() for _ in range(100)])
        return self.sample_metrics[sample_size]

    # @partial(jit, static_argnums=0) # TODO: replace np.repeat with smth else so I can use jit here.
    def kl_divergence(self, sample):
        """Kullback-Leibler divergence D(sample || p) between sample and distribution of class instance.

        Parameters
        ----------
        sample : array-like, shape (n,). Scalar-valued sample from some distribution.
        """
        histogram_likelihoods = utils.get_histogram_likelihoods(sample)
        return np.mean(np.log(histogram_likelihoods) - vmap(self.logpdf)(sample))

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        """
        Possible input shapes for mean and cov:
        1) mean.shape defines dimension of domain: if mean has shape (d,),
        then particles have shape (d,)
        2) if covariance is a scalar, it is reshaped to diag(3 * (cov,))
        3) if covariance is an array of shape (k,), it is reshaped to diag(cov)"""

        self.mean, self.cov = self._check_and_reshape_args(mean, cov)
        self.d = len(self.mean)
        self.expectations = self.compute_expectations(self.mean, self.cov)
        self.key = random.PRNGKey(0)
        self.sample_metrics = dict()
        self.initialize_metric_names()

    def _check_and_reshape_args(self, mean, cov):
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        if mean.ndim == 0:
            mean = np.reshape(mean, newshape=(1,))
        elif mean.ndim > 1:
            raise ValueError(f"Recieved inappropriate shape {mean.shape} for"
            "mean. (Wrong nr of dimensions).")
        d = mean.shape[0]

        if cov.ndim == 0:
            cov = d * (cov,)
            cov = np.asarray(cov)
            cov = np.diag(cov)
        elif cov.ndim == 1:
            assert len(cov) == len(mean)
            cov = np.diag(cov)
        assert mean.ndim == 1 and cov.ndim == 2
        assert mean.shape[0] == cov.shape[0] == cov.shape[1]
        if not utils.is_pd(cov):
            raise ValueError("Covariance must be positive definite.")
        return mean, cov

    def compute_expectations(self, mean, cov):
        """
        returns a list of expected values of the following expressions:
        x, x^2, cos(x), sin(x)
        """
        # characteristic function at [1, ..., 1]:
        t = np.ones(self.d)
        char = np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2)))
        expectations = [mean, np.diagonal(cov) + mean**2, np.real(char), np.imag(char)]
        return expectations

    def sample(self, n_samples):
        """mutates self.rkey"""
        self.newkey()
        out = random.multivariate_normal(self.key, self.mean, self.cov, shape=(n_samples,))
        self.newkey()

        shape = (n_samples, self.d)
        return out.reshape(shape)

    def logpdf(self, x):
        x = self._checkx(x)
        return stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def pdf(self, x):
        x = self._checkx(x)
        return stats.multivariate_normal.pdf(x, self.mean, self.cov)


class GaussianMixture(Distribution):
    def __init__(self, means, covs, weights):
        """
        Arguments:
        means, covs are np arrays or lists of length k, with entries of shape
        (d,) and (d, d) respectively. (e.g. covs can be array of shape (k, d, d))
        """
        means, covs, weights = self._check_and_reshape_args(means, covs, weights)
        self.d = len(means[0])
        self.expectations = self.compute_expectations(means, covs, weights)
        self.mean = self.expectations[0]
        # recall Cov(X) = E[XX^T] - mu mu^T =
        # sum_over_components(Cov(Xi) + mui mui^T) - mu mu^T
        mumut = np.einsum("ki,kj->kij", means, means) # shape (k, d, d)
        self.cov = np.average(covs + mumut, weights=weights, axis=0) \
                 - np.outer(self.mean, self.mean)
        self.key = random.PRNGKey(0)
        self.means = means
        self.covs = covs
        self.weights = weights
        self.num_components = len(weights)
        self.sample_metrics = dict()
        self.initialize_metric_names()

    def _check_and_reshape_args(self, means, covs, weights):
        means = np.asarray(means)
        covs = np.asarray(covs)
        weights = np.asarray(weights)
        weights = weights / np.sum(weights) # normalize
        assert len(means) == len(covs)
        if means.ndim == 1:
            means = means[:, np.newaxis]
        d = means.shape[1]
        if covs.ndim == 1 or covs.ndim == 2:
            covs = [np.identity(d) * var for var in covs]
            covs = np.array(covs)

        assert weights.ndim == 1 and len(weights) > 1
        assert means.ndim == 2 and covs.ndim == 3
        assert means.shape[1] == covs.shape[1] == covs.shape[2]

        for cov in covs:
            if not utils.is_pd(cov):
                raise ValueError("Covariance must be positive definite.")
        return means, covs, weights

    def compute_expectations(self, means, covs, weights):
        """
        returns a list of expected values of the following expressions:
        x, x^2, cos(x), sin(x)
        """
        # characteristic function at [1, ..., 1]:
        t = np.ones(self.d)
        chars = np.array([np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2))) for mean, cov in zip(means, covs)]) # shape (k,d)
        char = np.vdot(weights, chars)
        mean = np.einsum("i,id->d", weights, means)
        xsquares = [np.diagonal(cov) + mean**2 for mean, cov in zip(means, covs)]
        expectations = [mean, np.einsum("i,id->d", weights, xsquares),
                        np.real(char), np.imag(char)]
        expectations = [np.squeeze(e) for e in expectations]
        return expectations

    def sample(self, n_samples):
        """mutates self.rkey"""
        def sample_from_component(rkey, component, num_samples):
            return random.multivariate_normal(rkey,
                                              self.means[component],
                                              self.covs[component],
                                              shape=(num_samples,))
        components = random.categorical(self.key,
                                        np.log(self.weights),
                                        shape=(n_samples,))
        counts = onp.bincount(components.flatten())
        self.newkey()

        out = [
            sample_from_component(key, c, num)
            for key, c, num in
            zip(random.split(self.key, self.num_components), range(self.num_components), counts)]
        out = np.concatenate(out)
        self.newkey()

        shape = (n_samples, self.d)
        return out.reshape(shape)

    def _logpdf(self, x):
        """unnormalized"""
        x = np.array(x)
        if x.shape != (self.d,):
            raise ValueError(f"Input x must be an np.array of length {self.d} and dimension one.")

        def exponent(x, mean, cov):
            sigmax = np.dot(np.linalg.inv(cov), (x - mean))
            return - np.vdot((x - mean), sigmax) / 2
        exponents = vmap(exponent, (None, 0, 0))(x, self.means, self.covs) + np.log(self.weights)

        out = special.logsumexp(exponents)
        return np.squeeze(out)

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def pdf(self, x):
        x = np.asarray(x)
        if x.shape != (self.d,) and not (self.d == 1 and x.ndim == 0):
            raise ValueError(f"Input x must be an np.array of length {self.d} and dimension one.")
        pdfs = vmap(stats.multivariate_normal.pdf, (None, 0, 0))(x, self.means, self.covs)
        return np.vdot(pdfs, self.weights)

class Funnel(Distribution):
    def __init__(self, d):
        self.d = d
        self.mean = np.zeros(d)

        self.xcov = np.eye(d-1) * np.exp(9/2)
        self.ycov = 9
        self.cov = np.block([
            [self.xcov,          np.zeros((d-1, 1))],
            [np.zeros((1, d-1)), self.ycov         ]
        ])

    def sample(self, n_samples):
        self.newkey()
        y = random.normal(self.key, (n_samples, 1)) * 3
        self.newkey()
        x = random.normal(self.key, (n_samples, self.d-1)) * np.exp(y/2)
        return np.concatenate([x, y], axis=1)

    def pdf(self, x):
        x = self._checkx(x)
        *x, y = x
        x, y = np.asarray(x), np.asarray(y)

        xmean = np.zeros(self.d-1)
        xcov  = np.eye(self.d-1)*np.exp(y)
        py = stats.norm.pdf(y, loc=0, scale=3) # scale=stddev
        px = stats.multivariate_normal.pdf(x, mean=xmean, cov=xcov)
        return np.squeeze(py*px)

    def logpdf(self, x):
        x = self._checkx(x)
        *x, y = x
        x, y = np.asarray(x), np.asarray(y)
        xmean = np.zeros(self.d-1)
        xcov  = np.eye(self.d-1)*np.exp(y) # Cov(X \given Y=y)
        logpy = stats.norm.logpdf(y, loc=0, scale=3)
        logpx = stats.multivariate_normal.logpdf(x, mean=xmean, cov=xcov)
        return np.squeeze(logpy + logpx)

class FunnelizedGaussian(Gaussian):
    def __init__(self, mean, cov):
        self.mean, self.cov = self._check_and_reshape_args(mean, cov)
        self.d = len(self.mean)
        self.key = random.PRNGKey(0)

    def _check_and_reshape_args(self, mean, cov):
        if len(mean) < 2:
            raise ValueError("Funnel exists only in dimensions > 2."
            f"Received dimension len(mean) = {len(mean)}")
        return super()._check_and_reshape_args(mean, cov)

    def funnelize(self, v):
        """If v is standard 2D normal, then
        funnelize(v) is distributed as Neal's Funnel."""
        *x, y = v
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x*np.exp(3*y/2), 3*y)

    def defunnelize(self, z):
        """Inverse of funnelize."""
        *x, y = z
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x*np.exp(-y/2), y/3)

    def logpdf(self, z):
        x = self.defunnelize(z)
        *_, y = x
        return super().logpdf(x) + 3 * np.exp(3/2 * y)

    def pdf(self, z):
        x = self.defunnelize(z)
        *_, y = x
        return super().pdf(x) * 3 * np.exp(3/2 * y)

    def sample(self, n_samples):
        return vmap(self.funnelize)(super().sample(n_samples))

### Some nice target distributions
# 2D
l = np.asarray((1, 2, 1.5, 3, 3.3, 3.8))
l = onp.concatenate([-l, [0], l])
means = list(zip(l, (l**2)**0.8))
variances = [[1,1]]*len(means)
weights = [1]*len(means)
#bent = GaussianMixture(means, variances, weights)
bent_args = [means, variances, weights]
