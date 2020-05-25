import jax.numpy as np
from jax import grad, jit, vmap, random, jacfwd, jacrev
from jax.ops import index_update, index
from jax.scipy import stats, special

import numpy as onp

import svgd
import utils

def stein_operator(fun, x, logp, transposed=False):
    """
    Arguments:
    * fun: callable, transformation $\text{fun}: \mathbb R^d \to \mathbb R^d$, or $\text{fun}: \mathbb R^d \to \mathbb R$. Satisfies $\lim_{x \to \infty} \text{fun}(x) = 0$.
    * x: np.array of shape (d,).
    * p: callable, takes argument of shape (d,). Computes log(p(x)). Can be unnormalized (just using gradient.)

    Returns:
    Stein operator $\mathcal A$ evaluated at fun and x:
    \[ \mathcal A_p [\text{fun}](x) .\]
    This expression takes the form of a scalar if transposed else a dxd matrix
    """
    x = np.array(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"x needs to be an np.array of shape (d,). Instead, x has shape {x.shape}")
    fx = fun(x)
    if transposed:
        if fx.ndim == 0:   # f: R^d --> R
            raise ValueError(f"Got passed transposed = True, but the input function {fun.__name__} returns a scalar. This doesn't make sense: the transposed Stein operator acts only on vector-valued functions.")
        elif fx.ndim == 1: # f: R^d --> R^d
            return np.inner(grad(logp)(x), fun(x)) + np.trace(jacfwd(fun)(x).transpose())
        else:
            raise ValueError(f"Output of input function {fun.__name__} needs to have dimension 1 or two. Instead got output of shape {fx.shape}")
    else:
        if fx.ndim == 0:   # f: R^d --> R
            return grad(logp)(x) * fun(x) + grad(fun)(x)
        elif fx.ndim == 1: # f: R^d --> R^d
            return np.einsum("i,j->ij", grad(logp)(x), fun(x)) + jacfwd(fun)(x).transpose()
        elif fx.ndim == 2 and fx.shape[0] == fx.shape[1]:
            raise NotImplementedError()
#            return np.einsum("ij,j->ij", fun(x), grad(logp)(x)) + #np.einsum("iii->i", jacfwd(fun)(x).transpose())
        else:
            raise ValueError(f"Output of input function {fun.__name__} needs to be a scalar, a vector, or a square matrix. Instead got output of shape {fx.shape}")
            raise ValueError()


def ksd_squared(xs, logp, logh):
    """
    Arguments:
    * xs: np.array of shape (n, d)
    * logp: callable
    * k: callable, computes scalar-valued kernel k(x, y) given two input arguments.

    Returns:
    The square of the stein discrepancy KSD(q, p). Here, q is the empirical dist of xs.
    """
    k = lambda x, y: utils.ard(x, y, logh)
    def g(x, y):
        """x, y: np.arrays of shape (d,)"""
        def inner(x):
            kx = lambda y_: k(x, y_)
            return stein_operator(kx, y, logp)
        return stein_operator(inner, x, logp, transposed=True)

    gv  = vmap(g,  (0, None))
    gvv = vmap(gv, (None, 0))
    ksd_matrix = gvv(xs, xs)

    n = xs.shape[0]
    trace_indices = [list(range(n))]*2
    ksd_matrix = index_update(ksd_matrix, trace_indices, 0)

    return np.mean(ksd_matrix)
ksd_squared = jit(ksd_squared, static_argnums=1)

def phistar(xs, logp, logh, xest=None):
    if xest is not None:
        raise NotImplementedError()

    def f(x, y):
        """evaluated inside the expectation"""
        kx = lambda y: utils.ard(x, y, logh)
        return stein_operator(kx, y, logp, transposed=False)

    fv  = vmap(f,  (None, 0))
    fvv = vmap(fv, (0, None))
    phi_matrix = fvv(xs, xs)

    n = xs.shape[0]
    trace_indices = [list(range(n))]*2
    phi_matrix = index_update(phi_matrix, trace_indices, 0)

    return np.mean(phi_matrix, axis=1)

def stein(fun, xs, logp, transposed=False):
    """
    Arguments:
    * fun: callable, transformation fun: R^d \to R^d. Satisfies lim fun(x) = 0 for x \to \infty.
    * xs: np.array of shape (n, d). Used to compute an empirical distribution \hat q.
    * p: callable, takes argument of shape (d,). Computes log(p(x)). Can be unnormalized (just using gradient.)

    Returns:
    \[1/n \sum_i \mathcal A_p [\text{fun}](x) \]
    np.array of shape (d,) if transposed else shape (d, d)
    """
    return np.mean(vmap(stein_operator, (None, 0, None, None))(fun, xs, logp, transposed), axis=0)
