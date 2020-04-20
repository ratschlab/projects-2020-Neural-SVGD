import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd, jacrev
from jax.ops import index_update, index

import svgd

from tqdm import tqdm

# Kernelized Stein Discrepancy
def ksd(x, logp, bandwidth):
    """
    Arguments:
    * x: np.array of shape (n, d)
    * logp: callable
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
