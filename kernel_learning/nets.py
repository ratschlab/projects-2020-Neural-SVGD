import jax.numpy as np
from jax import vmap
import jax
import haiku as hk

import warnings
import utils

class RBF(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x, y):
        logh_init = np.zeros
        bandwidth = hk.get_parameter(
            "bandwidth", shape=x.shape, dtype=x.dtype, init=logh_init)
        return kernels.get_rbf_kernel(bandwidth)(x, y)

def build_rbf():
    def vanilla_rbf(x, y):
        rbf = RBF()
        return(rbf(x, y))
    return hk.transform(vanilla_rbf)

def build_mlp(sizes, name=None, skip_connection=False,
              with_bias=True, activate_final=False):
    """
    * sizes is a list of integers representing layer dimension

    Network uses He initalization; see https://github.com/deepmind/dm-haiku/issues/6
    and https://sonnet.readthedocs.io/en/latest/api.html#variancescaling.
    """
    def mlp(x):
        lin = hk.nets.MLP(output_sizes=sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=activate_final,
                          with_bias=with_bias,
                          name=name)
        if skip_connection is False:
            return lin(x)
        else:
            return lin(x) + x # make sure sizes fit (ie sizes[-1] == input dimension)
    return hk.transform(mlp)
