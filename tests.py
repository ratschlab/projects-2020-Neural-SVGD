import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

from utils import normsq, rbf, single_rbf


# kernel sanity check:
key = random.PRNGKey(0)
x = random.normal(key, (10,2))
h = 1

a = rbf(x, h)[0]

for i in range(10):
    assert (single_rbf(x[0], x[i], h) == a[i])


##############################
## update function testing
##########################
# experiment 1
from jax.scipy.stats import multivariate_normal
x = np.array([0., 1.])

def kernel(x, y):
    return single_rbf(x, y, h = 1)

def p(x):
    x = np.array(x)
    if len(x.shape) == 0:
        n = 1
    else:
        n = x.shape[0]
    return multivariate_normal.pdf(x, mean=np.zeros(n), cov=np.identity(n))

def logp(x):
    return np.log(p(x))

stepsize = 1

# compute update
xnew = update(x, logp, stepsize, kernel)


# check
check = np.array([- 2 / np.sqrt(np.e)]) # dunno what the second thing is
assert check[0] == xnew[0]

# experiment 2
x = np.array([1.])
stepsize = 0.5

xnew = update(x, logp, stepsize, kernel)
check = np.array([0.5])

assert xnew[0] == check[0]

# experiment 3 (just check convergence)
x = np.array([[0., 1.], [1, 1], [0, 0], [4, 5]])
stepsize = 0.1
x

for _ in range(20):
    xnew = update(x, logp, stepsize, kernel)
    diff = normsq(x - xnew)
    x = xnew

print(diff)
assert diff < 0.1





