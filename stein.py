from functools import partial
import jax.numpy as np
from jax import grad, vmap, random, jacfwd, jit
from jax.ops import index_update, index

def stein_operator(fun, x, logp, transposed=False, aux=False):
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
            auxdata = None
            out = np.inner(grad(logp)(x), fun(x)) + np.trace(jacfwd(fun)(x).transpose())
        else:
            raise ValueError(f"Output of input function {fun.__name__} needs to have dimension 1 or two. Instead got output of shape {fx.shape}")
    else:
        if fx.ndim == 0:   # f: R^d --> R
            drift_term = grad(logp)(x) * fx
            repulsive_term = grad(fun)(x)
            auxdata = np.asarray([drift_term, repulsive_term])
            out = drift_term + repulsive_term
        elif fx.ndim == 1: # f: R^d --> R^d
            auxdata = None
            out = np.einsum("i,j->ij", grad(logp)(x), fun(x)) + jacfwd(fun)(x).transpose()
        elif fx.ndim == 2 and fx.shape[0] == fx.shape[1]:
            raise NotImplementedError()
#            return np.einsum("ij,j->ij", fun(x), grad(logp)(x)) + #np.einsum("iii->i", jacfwd(fun)(x).transpose())
        else:
            raise ValueError(f"Output of input function {fun.__name__} needs to be a scalar, a vector, or a square matrix. Instead got output of shape {fx.shape}")
    if aux:
        return out, auxdata
    else:
        return out

def stein(fun, xs, logp, transposed=False, aux=False):
    """
    Arguments:
    * fun: callable, transformation fun: R^d \to R^d. Satisfies lim fun(x) = 0 for x \to \infty.
    * xs: np.array of shape (n, d). Used to compute an empirical distribution \hat q.
    * p: callable, takes argument of shape (d,). Computes log(p(x)). Can be unnormalized (just using gradient.)

    Returns: the expectation of the Stein operator $\mathcal A [\text{fun}]$ wrt the empirical distribution of the particles xs:
    \[1/n \sum_i \mathcal A_p [\text{fun}](x_i) \]
    np.array of shape (d,) if transposed else shape (d, d)
    """
    if aux:
        steins, auxdata = vmap(stein_operator, (None, 0, None, None, None))(fun, xs, logp, transposed, aux)
        return np.mean(steins, axis=0), np.mean(auxdata, axis=0) # per-particle drift and repulsion, shape (2, d)
    else:
        steins = vmap(stein_operator, (None, 0, None, None, None))(fun, xs, logp, transposed, aux)
        return np.mean(steins, axis=0)

def phistar_i(xi, x, logp, kernel):
    """
    Arguments:
    * xi: np.array of shape (d,), usually a row element of x
    * x: np.array of shape (n, d)
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y.

    Returns:
    * \phi^*(xi) estimated using the particles x
    * auxdata consisting of [mean_drift, mean_repulsion] of shape (2, d)
    """
    if xi.ndim > 1:
        raise ValueError(f"Shape of xi must be (d,). Instead, received shape {xi.shape}")
    kx = lambda y: kernel(y, xi)
    return stein(kx, x, logp, aux=True)

def phistar(followers, leaders, logp, kernel):
    """
    Returns an np.array of shape (n, d) containing values of phi^*(x_i) for i in {1, ..., n}.

    Arguments:
    * follower: np.array of shape (n, d)
    * leader: np.array of shape (l, d). Usually a subsample l < n of particles.
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y.

    Returns:
    np array of same shape as x.
    """
    return vmap(phistar_i, (0, None, None, None))(followers, leaders, logp, kernel)

# def phistar(xs, logp, k, xest=None):
#     if xest is not None:
#         raise NotImplementedError()
#     def f(x, y):
#         """evaluated inside the expectation"""
#         kx = lambda y: k(x, y)
#         return stein_operator(kx, y, logp, transposed=False)
#
#     fv  = vmap(f,  (None, 0))
#     fvv = vmap(fv, (0, None))
#     phi_matrix = fvv(xs, xs)
#
#     n = xs.shape[0]
#     trace_indices = [list(range(n))]*2
#     phi_matrix = index_update(phi_matrix, trace_indices, 0)
#
#     return np.mean(phi_matrix, axis=1)

@partial(jit, static_argnums=(1,2))
def ksd_squared_u(xs, logp, k):
    """
    U-statistic for KSD^2. Computation in O(n^2)
    Arguments:
    * xs: np.array of shape (n, d)
    * logp: callable
    * k: callable, computes scalar-valued kernel k(x, y) given two input arguments.

    Returns:
    The square of the stein discrepancy KSD(q, p).
    KSD is approximated as $1 / n(n-1) \sum_{i \neq j} g(x_i, x_j)$, where the x are iid distributed as q
    """
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
    diagonal_indices = [list(range(n))]*2
    ksd_matrix = index_update(ksd_matrix, diagonal_indices, 0)

    return np.sum(ksd_matrix) / (n * (n-1))

@partial(jit, static_argnums=(2,3))
def ksd_squared(xs, ys, logp, k):
    """
    O(n*m)
    Arguments:
    * xs: np.array of shape (n, d)
    * ys: np.array of shape (m, d) (can be the same array as xs)
    * logp: callable
    * k: callable, computes scalar-valued kernel k(x, y) given two input arguments.

    Returns:
    The square of the stein discrepancy KSD(q, p).
    KSD is approximated as $\sum_i \sum_j g(x_i, y_j)$, where the x and y are iid distributed as q
    """
    def g(x, y):
        """x, y: np.arrays of shape (d,)"""
        def inner(x):
            kx = lambda y_: k(x, y_)
            return stein_operator(kx, y, logp)
        return stein_operator(inner, x, logp, transposed=True)
    gv  = vmap(g,  (0, None))
    gvv = vmap(gv, (None, 0))
    ksd_matrix = gvv(xs, ys)
    return np.mean(ksd_matrix)

@partial(jit, static_argnums=(2,3,4))
def ksd_squared_l(xs, ys, logp, k, return_variance=False):
    """
    O(n) time estimator for the KSD.
    Arguments:
    * xs: np.array of shape (n, d)
    * ys: np.array of shape (n, d) (can be the same array as xs)
    * logp: callable
    * k: callable, computes scalar-valued kernel k(x, y) given two input arguments of shape (d,).

    Returns:
    * The square of the stein discrepancy KSD(q, p).
    KSD is approximated as $\sum_i g(x_i, y_i)$, where the x and y are iid distributed as q
    * The approximate variance of h(X, Y)
    """
    if xs.shape != ys.shape: raise ValueError(f"xs and ys must have same shape. Instead, received shapes {xs.shape} and {ys.shape}.")
    def h(x, y):
        """x, y: np.arrays of shape (d,)"""
        def inner(x):
            kx = lambda y_: k(x, y_)
            return stein_operator(kx, y, logp)
        return stein_operator(inner, x, logp, transposed=True)
    outs = vmap(h)(xs, ys)
    if return_variance:
        return np.mean(outs), np.var(outs, ddof=1) / xs.shape[0]
    else:
        return np.mean(outs)

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
