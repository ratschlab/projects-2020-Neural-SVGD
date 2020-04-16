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


metric_names = ["MSE X", "MSE X^2", "KSD 0.1", "KSD 1", "KSD 10"]
def get_metrics(xout, logp):
    """
    Compute metrics. One-dim case. Target p is gaussian mixture.
    """
    assert xout.ndim == 2
    assert xout.shape[1] == 1
    assert not np.any(np.isnan(xout))

    metrics = []
    metrics.extend(mses(xout))
    for ksd_bandwidth in [0.1, 1, 10]:
        metrics.append(ksd(xout, logp, ksd_bandwidth))
    return metrics
