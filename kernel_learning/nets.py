import jax.numpy as np
from jax import vmap
import jax
import haiku as hk

import warnings
import utils
import kernels
from typing import Any, Callable, Iterable, Optional, Type

def bandwidth_init(shape, dtype=np.float32):
    """Init for bandwith matrix"""
    d = shape[0]
    return np.identity(d, dtype)

class RBFKernel(hk.Module):
    def __init__(self, scale_param=False, parametrization="diagonal", name=None):
        """
        * If params='diagonal', use one scalar bandwidth parameter per dimension,
        i.e. parameters habe shape (d,).
        * If params=log_diagonal, same but parametrize log(bandwidth)
        * If params='full', parametrize kernel using full (d, d) matrix.
        Params are initialized st the two options are equivalent at initialization."""
        super().__init__(name=name)
        self.parametrization = parametrization
        self.scale_param = scale_param

    def __call__(self, xy):
        """xy should have shape (2, d)"""
        d = xy.shape[-1]
        scale = hk.get_parameter("scale", shape=(), init=np.ones) if self.scale_param else 1.
        if self.parametrization == "log_diagonal":
            log_bandwidth = hk.get_parameter("log_bandwidth", shape=(d,), init=np.zeros)
            log_bandwidth = np.clip(log_bandwidth, a_min=-5, a_max=5)
            return scale * kernels.get_rbf_kernel_logscaled(log_bandwidth)(*xy)
        elif self.parametrization == "diagonal":
            bandwidth = hk.get_parameter("bandwidth", shape=(d,), init=np.ones)
            return scale * kernels.get_rbf_kernel(bandwidth)(*xy)
        elif self.parametrization == "full":
            sigma = hk.get_parameter("sigma", shape=(d, d), init=bandwidth_init)
            return scale * kernels.get_multivariate_gaussian_kernel(sigma)(*xy)

class DeepKernel(hk.Module):
    def __init__(self, sizes, name=None):
        super().__init__(name=name)
        self.sizes = sizes

    def __call__(self, x):
        """x should have shape (2, d)"""
        k = RBFKernel(scale_param=True, parametrization="full")
        net = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        return k(net(x))


def get_norm(init_x):
    mean = np.mean(init_x, axis=0)
    std = np.std(init_x, axis=0)
    def norm(x):
        return (x - mean) / (std + 1e-5)
    return norm


class VectorField(hk.Module):
    def __init__(self, sizes: list, name: str = None):
        """
        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes

    def __call__(self, x):
        """x is a batch of particles of shape (n, d) or a single particle
        of shape (d,)"""
        assert x.shape[-1] == self.sizes[-1]
        mlp = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        return mlp(x)


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
