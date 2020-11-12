import sys
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning")

import jax
from jax import grad, random
from jax import numpy as np
import haiku as hk

from typing import Any, Generator, Mapping, Tuple

import metrics, utils, stein, kernels, train

rkey = random.PRNGKey(0)


dist = distributions.Gaussian(0, 1)


def net_fn(x):
    mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(2),
    ])
    return mlp(x)

net = hk.transform(net_fn)
x = dist.sample((1, 10))
params = net.init(rkey, x)
del net

net = hk.transform(net_fn)
x = dist.sample((3, 10))
output = net.apply(params, x)

x = dist.sample((10,))
output = net.apply(params, x)

output


params["linear"]["w"].shape





x.shape


net = kernels.linear_regression





batch = np.asarray([dist.sample((10, 1)) for _ in range(2)], dtype=np.float32)


batch = x


params = net.init(rkey, batch)

output.shape


params



