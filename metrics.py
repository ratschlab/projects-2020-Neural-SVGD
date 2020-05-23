import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd, jacrev
from jax.ops import index_update, index
from jax.scipy import stats, special

import numpy as onp

import svgd
import utils
import stein

# distributions packaged with metrics
# check wikipedia for computation of higher moments https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Higher_moments
# also recall form of characteristic function https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)#Examples
class Distribution():
    """Base class for package logpdf + metrics + sampling"""
    def __init__(self):
        self.expectations = None
        self.sample_metrics = None
        self.d = None
        self.ksd_grid = (0.1, 1, 10)
        self.initialize_metric_names()
        pass

    key = random.PRNGKey(0)

    def newkey(self):
        self.key = random.split(self.key)[0]

    def sample(self, shape):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def compute_metrics(self, x, normalize=False):
        """Compute metrics given samples.
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
            square_errors = np.array(square_errors) # shape (4, d)

            ksds = [ksd_squared(x, self.logpdf, np.log(np.array(h))) for h in self.ksd_grid]
            ksds = np.array(ksds)

            metrics_dict = {
                "square_errors": square_errors, # shape (4, d)
                "ksds": ksds # shape (len(ksd_grid),)
            }
            return metrics_dict

    def get_metrics_shape(self):
        return {
        "square_errors": (4, self.d),
        "ksds": (len(self.ksd_grid),)
        }

    def initialize_metric_names(self):
        self.metric_names = {
            "square_errors": [f"SE for {val}" for val in ["X", "X^2", "cos(X)", "sin(X)"]],
            "ksds": [f"KSD for h = {h}" for h in self.ksd_grid]
        }
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

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        """self.expectations is a list of expected values of the following expressions: x, x^2, cos(x), sin(x)"""
        mean = np.array(mean)
        cov = np.array(cov)
        if cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 0:
            cov = cov[np.newaxis, np.newaxis]
        if mean.ndim == 0:
            mean = np.ones(len(cov)) * mean
        assert mean.ndim == 1 and cov.ndim == 2
        assert mean.shape[0] == cov.shape[0] == cov.shape[1]

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
        self.sample_metrics = dict()

        self.ksd_grid = (0.1, 1, 10)
        self.initialize_metric_names()
        return None

    def sample(self, shape):
        self.newkey()
        out = random.multivariate_normal(self.key, self.mean, self.cov, shape)
        self.newkey()
        return out.reshape(shape + (self.d,)) if self.d > 1 else out.reshape(shape)

    def _checkx(self, x):
        """check if particle (single particle shape (d,)) in right shape, etc"""
        x = np.array(x)
        if x.ndim == 0 and self.d == 1:
            x = x.reshape((1,))
        if x.shape != (self.d,):
            raise ValueError(f"x needs to have shape ({self.d},). Instead, received x of shape {x.shape}.")
        return x

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
        means, covs are np arrays or lists of length k, with entries of shape (d,) and (d, d) respectively. (e.g. covs can be array of shape (k, d, d))

        Initialization:
        self.expectations is a list of expected values of the following expressions: x, x^2, cos(x), sin(x)"""
        means = np.array(means)
        covs = np.array(covs)
        weights = np.array(weights)
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

        # characteristic function at [1, ..., 1]:
        t = np.ones(d)
        chars = np.array([np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2))) for mean, cov in zip(means, covs)]) # shape (k,d)
        char = np.vdot(weights, chars)

        mean = np.einsum("i,id->d", weights, means)
        xsquares = [np.diagonal(cov) + mean**2 for mean, cov in zip(means, covs)]
        self.expectations = [mean, np.einsum("i,id->d", weights, xsquares), np.real(char), np.imag(char)]
        self.expectations = [np.squeeze(e) for e in self.expectations]

        self.key = random.PRNGKey(0)
        self.means = means
        self.covs = covs
        self.weights = weights
        self.num_components = len(weights)
        self.d = d # particle dimension
        self.mean = mean
        # recall Cov(X) = E[XX^T] - mu mu^T = sum_over_components(Cov(Xi) + mui mui^T) - mu mu^T
        mumut = np.einsum("ki,kj->kij", means, means) # shape (k, d, d)
        self.cov = np.average(covs + mumut, weights=weights, axis=0) - np.outer(mean, mean)
        self.sample_metrics = dict()

        self.ksd_grid = (0.1, 1, 10)
        self.initialize_metric_names()
        return None

    def sample(self, shape):
        """mutates self.rkey"""
        def sample_from_component(rkey, component, num_samples):
            return random.multivariate_normal(rkey, self.means[component], self.covs[component], shape=(num_samples,))
        components = random.categorical(self.key, np.log(self.weights), shape=shape)
        counts = onp.bincount(components.flatten())
        self.newkey()

        out = [sample_from_component(key, c, num) for key, c, num in zip(random.split(self.key, self.num_components), range(self.num_components), counts)]
        out = np.concatenate(out)
        self.newkey()

        return out.reshape(shape + (self.d,)) if self.d > 1 else out.reshape(shape)

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
        x = np.array(x)
        if x.shape != (self.d,) and not (self.d == 1 and x.ndim == 0):
            raise ValueError(f"Input x must be an np.array of length {self.d} and dimension one.")
        pdfs = vmap(stats.multivariate_normal.pdf, (None, 0, 0))(x, self.means, self.covs)
        return np.log(np.vdot(pdfs, self.weights))

    def pdf(self, x):
        x = np.array(x)
        if x.shape != (self.d,) and not (self.d == 1 and x.ndim == 0):
            raise ValueError(f"Input x must be an np.array of length {self.d} and dimension one.")
        pdfs = vmap(stats.multivariate_normal.pdf, (None, 0, 0))(x, self.means, self.covs)
        return np.vdot(pdfs, self.weights)

######################################
# Kernelized Stein Discrepancy
def _ksd(x, logp, bandwidth):
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

    return np.mean(vmap(ksd_i, (0, None, None, None))(x, x, logp, bandwidth)) / x.shape[0]


def ksd_squared(xs, logp, bandwidth):
    kernel = lambda x, y: utils.ard(x, y, bandwidth)
    def phistar(z):
        """z has shape (d,).
        Returns array of shape (d,)."""
        k = lambda x: kernel(x, z)
        return stein.stein(k, xs, logp)
    return stein.stein(phistar, xs, logp, transposed=True)
# NOTE: this is actually the square of the stein discrepancy.

ksd_squared = jit(ksd_squared, static_argnums=1)

########################
### metrics to log while running SVGD
def initialize_log(self):
    """self is an SVGD object"""
    d = self.particle_shape[1]
    log = {
        "desc": dict(),
        "metrics": dict()
    }
    log["desc"] = {
        "particle_mean": np.zeros(shape=(self.n_iter_max, d)),
        "particle_var":  np.zeros(shape=(self.n_iter_max, d)),
        "bandwidth":     np.zeros(shape=(self.n_iter_max, d))
    }

    for key, shape in self.dist.get_metrics_shape().items():
        log["metrics"][key] = np.zeros(shape=(self.n_iter_max,) + shape)

    return log


def update_log(self, i, x, log, bandwidth):
    """
    Input: log is a dict with two subdicts, "metrics" and "desc". This function updates these dicts.
    1) create dict of updates
    2) 'append' dict of updates to log dict
    """
    bandwidth = np.exp(bandwidth) # cause else h is rescaled to log(h)
    update_dict = {
        "desc": dict(),
        "metrics": dict()
    }
    update_dict["desc"] = {
        "particle_mean": np.mean(x, axis=0),
        "particle_var": np.var(x, axis=0),
        "bandwidth": bandwidth
    }

    # metrics
    update_dict["metrics"] = self.dist.compute_metrics(x)

    for key in log.keys():
        if key == "metrics":
            for k, v in log[key].items():
                log["metrics"][k] = index_update(v, index[i, ...], update_dict["metrics"][k])
        elif key == "desc":
            for k, v in log[key].items():
                log[key][k] = index_update(v, index[i, :], update_dict[key][k])
        else:
            pass
    return log
