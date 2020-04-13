import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd, jacrev
from jax.ops import index_update, index

import svgd

# Kernelized Stein Discrepancy
def ksd_i(xi, x, logp, bandwidth):
    dphi_dxi_transposed = lambda xi: np.trace(jacfwd(svgd.phistar_i)(xi, x, logp, bandwidth))
    return np.vdot(grad(logp)(xi), svgd.phistar_i(xi, x, logp, bandwidth)) + dphi_dxi_transposed(xi)

def ksd(x, logp, bandwidth):
    return np.mean(vmap(ksd_i, (0, None, None, None))(x, x, logp, bandwidth))
