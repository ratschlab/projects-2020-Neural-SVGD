import jax.numpy as np
from jax import vmap
import jax
import haiku as hk

import warnings
import utils

class ARD(hk.Module):
    def __init__(self, name=None):
        super(ARD, self).__init__(name=name)

    def __call__(self, x, y):
        logh_init = np.zeros
        logh = hk.get_parameter("logh", shape=x.shape, dtype=x.dtype, init=logh_init)
        return kernels.get_rbf_kernel_logscaled(logh)(x, y)

def vanilla_ard(x, y):
    ard = ARD()
    return(ard(x, y))

def build_mlp(sizes, name=None, skip_connection=False):
    """
    * sizes is a list of integers representing layer dimension

    Network uses He initalization; see https://github.com/deepmind/dm-haiku/issues/6
    and https://sonnet.readthedocs.io/en/latest/api.html#variancescaling.
    """
    def mlp(x):
        lin = hk.nets.MLP(output_sizes=sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.relu,
                          activate_final=False,
                          name=name)
        if skip_connection is False:
            return lin(x)
        else:
            return lin(x) + x # make sure sizes fit (ie sizes[-1] == input dimension)
    return hk.transform(mlp)
