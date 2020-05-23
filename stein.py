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
    np.array of shape (d,) if transposed else (d, d) # TODO check if shapes are correct
    """
    x = np.array(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"x needs to be an np.array of shape (d,). Instead, x has shape {x.shape}")
    if transposed:
        out = stein_operator(fun, x, logp, transposed=False)
        if out.ndim == 1 and x.shape[0] != 1:
            raise ValueError(f"Got passed transposed = True, but the input function {fun.__name__} returns a scalar. This doesn't make sense: the transposed Stein operator acts on functions in R^d.")
        elif out.ndim == 1:
            return out
        elif out.ndim == 2:
            return np.trace(out)
        else:
            raise ValueError("Output of stein_operator must have shape (d,) or (d, d)")
    else:
        fx = fun(x)
        if fx.ndim == 0:   # f: R^d --> R
            return grad(logp)(x) * fun(x) + grad(fun)(x)
        elif fx.ndim == 1: # f: R^d --> R^d
            return np.einsum("i,j->ij", grad(logp)(x), fun(x)) + jacfwd(fun)(x)
        else:
            raise ValueError()

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
