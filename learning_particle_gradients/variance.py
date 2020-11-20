from functools import partial
import jax.numpy as np
from jax import grad, vmap, random, jacfwd, jit
from jax.ops import index_update, index

import utils

def var_exp(gram):
    """
    Estimator for $Var_Y(E_X[h(X, Y)])$, where h is a symmetric kernel function,
    and X, Y are iid.
    For background, see this technical report: https://arxiv.org/pdf/1906.02104.pdf

    Args
        gram: Gram matrix $G_{ij} = h(x_i, x_j)$. Has shape (n, n)
    """
    if gram.shape[0] != gram.shape[1] or gram.ndim != 2:
        raise ValueError(f"Gram matrix must have shape (n, n). Instead received shape {gram.shape}.")

    n = gram.shape[0]
    diagonal_indices = [list(range(n))]*2
    gramzero = index_update(gram, diagonal_indices, 0)
    del gram
    one_n = np.ones(n)
    expectation_of_square = (np.linalg.norm(np.dot(gramzero, one_n))**2 \
                             - np.linalg.norm(gramzero)**2) / (n * (n-1) * (n-2))
    square_of_expectation = (utils.vmv_dot(one_n, gramzero, one_n)**2 \
                             - 4 * np.linalg.norm(np.dot(gramzero, one_n))**2 \
                             + 2 * np.linalg.norm(gramzero)**2) / (n*(n-1)*(n-1)*(n-3))
    return expectation_of_square - square_of_expectation


def var_hxy(gram):
    """
    Estimator for $Var_{XY}(h(X, Y))$, where h is a symmetric kernel function,
    and X, Y are iid.
    For background, see this technical report: https://arxiv.org/pdf/1906.02104.pdf

    Args
        gram: Gram matrix $G_{ij} = h(x_i, x_j)$. Has shape (n, n)
    """
    if gram.shape[0] != gram.shape[1] or gram.ndim != 2:
        raise ValueError(f"Gram matrix must have shape (n, n). Instead received shape {gram.shape}.")

    n = gram.shape[0]
    diagonal_indices = [list(range(n))]*2
    gramzero = index_update(gram, diagonal_indices, 0)

    ones = np.ones(n)
    mean_of_square = np.linalg.norm(gramzero)**2 / (n * (n-1))
    mean_squared = (np.dot(np.dot(ones, gramzero), ones)**2 \
                    - 4*np.linalg.norm(np.dot(gramzero, ones))**2 \
                    + 2*np.linalg.norm(gramzero)**2) / (n*(n-1)*(n-2)*(n-3))
    return mean_of_square - mean_squared


def var_ksd(gram):
    """
    Estimator for $Var_{XY} \hat KSD^2$, where \hat KSD^2 is a U-estimator for the
    squared kernelized Stein discrepancy with kernel k.
    For background, see this technical report: https://arxiv.org/pdf/1906.02104.pdf

    Args
        gram: Gram matrix $G_{ij} = h(x_i, x_j)$. Has shape (n, n)
    """
    n = gram.shape[0]
    return 4*(n-2) / (n * (n-1)) * var_exp(gram) + 2/(n*(n-1)) * var_hxy(gram)

def compute_var_ksd(xs, logp: callable, k: callable):
    """
    Estimator for $Var_{XY} \hat KSD$, where \hat KSD is a U-estimator for the
    squared kernelized Stein discrepancy with kernel k.
    For background, see this technical report: https://arxiv.org/pdf/1906.02104.pdf

    Args
        xs: random samples from q
        logp: computes log(p(x))
        k: computes positive definite symmetric kernel k(x, y)
    """
    def h(x, y):
        def h2(x_, y_): return np.inner(grad(logp)(y_), grad(k, argnums=0)(x_, y_))
        def d_xk(x_, y_): return grad(k, argnums=0)(x_, y_)
        out = np.inner(grad(logp)(x), grad(logp)(y)) * k(x,y) +\
                h2(x, y) + h2(y, x) +\
                np.trace(jacfwd(d_xk, argnums=1)(x, y))
        return out

    gram = vmap(vmap(h, (0, None)), (None, 0))(xs, xs)
    return var_ksd(gram)

@partial(jit, static_argnums=(2,3))
def h_var(xs, ys, logp, k):
    """Estimate for Var(h(X, Y))
    Recall Var(KSD_L) = 1/n Var(h(X, Y))"""
    def h(x, y):
        """x, y: np.arrays of shape (d,)"""
        def inner(x):
            kx = lambda y_: k(x, y_)
            return stein_operator(kx, y, logp)
        return stein_operator(inner, x, logp, transposed=True)
    return np.var(vmap(h)(xs, ys), ddof=1) # unbiased variance
