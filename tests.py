######### TESTS
## this script throws assertion errors if something weird is going on.
## TODO: read up on best practices for unit testing


import jax.numpy as np
from jax import grad, jit, vmap, random


#################################
##### pairwise distances
from scipy.spatial.distance import pdist, squareform
from utils import squared_distance_matrix

key = random.PRNGKey(0)
x1 = random.normal(key, shape=(20, 5))
key = random.PRNGKey(1)
x2 = random.normal(key, shape=(30, 1))

for x in [x1, x2]:
    assert np.all(np.absolute(squareform(pdist(x)**2) - (squared_distance_matrix(x))) < 0.01)

