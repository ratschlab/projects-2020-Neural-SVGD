import jax.numpy as jnp
from jax import vmap, grad
import jax
import haiku as hk
from . import kernels


def bandwidth_init(shape, dtype=jnp.float32):
    """Init for bandwith matrix"""
    d = shape[0]
    return jnp.identity(d, dtype)


class RBFKernel(hk.Module):
    def __init__(self, scale_param=False, parametrization="diagonal", name=None):
        """
        * If params='diagonal', use one scalar bandwidth parameter per dimension,
        i.e. parameters habe shape (d,).
        * If params=log_diagonal, same but parametrize log(bandwidth)
        * If params='full', parametrize kernel using full (d, d) matrix.
        Params are initialized st the three options are equivalent at initialization."""
        super().__init__(name=name)
        self.parametrization = parametrization
        self.scale_param = scale_param

    def __call__(self, xy):
        """xy should have shape (2, d)"""
        d = xy.shape[-1]
        scale = hk.get_parameter("scale", shape=(), init=jnp.ones) if self.scale_param else 1.
        if self.parametrization == "log_diagonal":
            log_bandwidth = hk.get_parameter("log_bandwidth", shape=(d,), init=jnp.zeros)
            log_bandwidth = jnp.clip(log_bandwidth, a_min=-5, a_max=5)
            return scale * kernels.get_rbf_kernel_logscaled(log_bandwidth)(*xy)
        elif self.parametrization == "diagonal":
            bandwidth = hk.get_parameter("bandwidth", shape=(d,), init=jnp.ones)
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
    mean = jnp.mean(init_x, axis=0)
    std = jnp.std(init_x, axis=0)

    def norm(x):
        return (x - mean) / (std + 1e-5)
    return norm


class MLP(hk.Module):
    def __init__(self, sizes: list, name: str = None):
        """
        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes

    def __call__(self, x: jnp.ndarray, dropout: bool = False):
        """
        args:
            x: a batch of particles of shape (n, d) or a single particle
        of shape (d,)
            dropout: bool; apply dropout to output?
        """
        mlp = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        output = mlp(x)
        if dropout:
            output = hk.dropout(
                rng=hk.next_rng_key(),
                rate=0.2,
                x=output
            )
        return output


class KLGrad(hk.Module):
    def __init__(self, sizes: list, logp: callable, name: str = None):
        """
        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes
        self.logp = logp

    def __call__(self, x):
        """x is a batch of particles of shape (n, d) or a single particle
        of shape (d,)"""
        assert x.shape[-1] == self.sizes[-1]
        mlp = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        s = hk.get_parameter("scale", (), init=jnp.ones)
        if x.ndim == 1:
            return mlp(x) - s*grad(self.logp)(x)
        elif x.ndim == 2:
            return mlp(x) - s*vmap(grad(self.logp))(x)
        else:
            raise ValueError("Input needs to have rank 1 or 2.")


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
            return lin(x) + x  # make sure sizes fit (ie sizes[-1] == input dimension)
    return hk.transform(mlp)


class StaticHypernet(hk.Module):
    def __init__(self, sizes: list = [256, 256], embedding_size=64, name: str = None):
        """
        args
            sizes: sizes of the fully connected layers
            embedding_size: dimension of z (input to hypernetwork)

        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes
        self.embedding_size = embedding_size

    def __call__(self, base_params: jnp.ndarray, dropout: bool = False):
        """
        args:
            base_params: parameters of the base network, layer-wise. Must have
                shape (m, _), where m is the number of layers in the base
                convnet.
            dropout: bool; apply dropout to output?
        """
        num_base_layers = len(base_params)
        mlp = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)

        z = hk.get_parameter("z",
                             shape=(num_base_layers, self.embedding_size),
                             init=hk.initializers.RandomNormal())

        output = mlp(jnp.hstack((base_params, z)))  # input now has (batched) shape
                                                    # (num_base_layers, self.embedding_size + layer_size)
        if dropout:
            output = hk.dropout(
                rng=hk.next_rng_key(),
                rate=0.2,
                x=output
            )
        return output


NUM_CLASSES = 10
initializer = hk.initializers.RandomNormal(stddev=1 / 100)

class CNN(hk.Module):
    def __init__(self, n_channels=8, n_classes=10, depth=2, name: str = None):
        super().__init__(name=name)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.initializer = hk.initializers.RandomNormal(stddev=1 / 100)

    def __call__(self, image): # TODO: output should have length self.n_classes
#        conv_layers = self.depth * [hk.Conv2D(self.n_channels,
#                                              kernel_shape=3,
#                                              w_init=self.initializer,
#                                              b_init=self.initializer,
#                                              stride=2),
#                                    jax.nn.relu]
#        convnet = hk.Sequential(conv_layers + [hk.Flatten()])

        convnet = hk.Sequential([
            hk.Conv2D(self.n_channels,
                      kernel_shape=3,
                      w_init=self.initializer,
                      b_init=self.initializer,
                      stride=2),
            jax.nn.relu,

            hk.Conv2D(self.n_channels,
                      kernel_shape=3,
                      w_init=self.initializer,
                      b_init=self.initializer,
                      stride=2),
            jax.nn.relu,

            hk.Flatten(),
        ])
        return convnet(image)