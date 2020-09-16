from functools import partial
import jax.numpy as np
from jax import grad, vmap, random, jacfwd, jit
from jax.ops import index_update, index

import variance
import distributions
import kernels

def stein_operator(fun, x, logp, transposed=False, aux=False):
    """
    Arguments:
    * fun: callable, transformation $\text{fun}: \mathbb R^d \to \mathbb R^d$,
    or $\text{fun}: \mathbb R^d \to \mathbb R$.
    Satisfies $\lim_{x \to \infty} \text{fun}(x) = 0$.
    * x: np.array of shape (d,).
    * p: callable, takes argument of shape (d,). Computes log(p(x)). Can be
    unnormalized (just using gradient.)

    Returns:
    Stein operator $\mathcal A$ evaluated at fun and x:
    \[ \mathcal A_p [\text{fun}](x) .\]
    This expression takes the form of a scalar if transposed else a dxd matrix

    Auxiliary data: values for G (kernel smoothed gradient) and R (kernel
    repulsion term) of shape((G, R)) = (2, d)
    """
    x = np.array(x, dtype=np.float32)
    #if x.ndim < 1: # assume d = 1
    #    x = np.expand_dims(x, 0) # x now has correct shape (d,) = (1,)
    if x.ndim != 1:
        raise ValueError(f"x needs to be an np.array of shape (d,). Instead, "
                         f"x has shape {x.shape}")
    fx = fun(x)
    if transposed:
        if fx.ndim == 0:   # f: R^d --> R
            raise ValueError(f"Got passed transposed = True, but the input "
                             f"function {fun.__name__} returns a scalar. This "
                             "doesn't make sense: the transposed Stein operator "
                             "acts only on vector-valued functions.")
        elif fx.ndim == 1: # f: R^d --> R^d
            drift_term = np.inner(grad(logp)(x), fx)
            repulsive_term = np.trace(jacfwd(fun)(x).transpose())
            auxdata = np.asarray([drift_term, repulsive_term])
            out = drift_term + repulsive_term
        else:
            raise ValueError(f"Output of input function {fun.__name__} needs "
                             f"to have rank 0 or 1. Instead got output "
                             f"of shape {fx.shape}")
    else:
        if fx.ndim == 0:   # f: R^d --> R
            drift_term = grad(logp)(x) * fx
            repulsive_term = grad(fun)(x)
            auxdata = np.asarray([drift_term, repulsive_term])
            out = drift_term + repulsive_term
        elif fx.ndim == 1: # f: R^d --> R^d
            drift_term = np.einsum("i,j->ij", grad(logp)(x), fun(x))
            repulsive_term = jacfwd(fun)(x).transpose()
            auxdata = np.asarray([drift_term, repulsive_term])
            out = drift_term + repulsive_term
        elif fx.ndim == 2 and fx.shape[0] == fx.shape[1]: # f: R^d --> R^{dxd}
            raise NotImplementedError("Not implemented for matrix-valued f.")
        else:
            raise ValueError(f"Output of input function {fun.__name__} needs "
                             f"to be a scalar, a vector, or a square matrix. "
                             f"Instead got output of shape {fx.shape}")
    if aux:
        return out, auxdata
    else:
        return out

def stein_expectation(fun, xs, logp, transposed=False, aux=False):
    """
    Arguments:
    * fun: callable, transformation fun: R^d \to R^d. Satisfies lim fun(x) = 0 for x \to \infty.
    * xs: np.array of shape (n, d). Used to compute an empirical distribution \hat q.
    * p: callable, takes argument of shape (d,). Computes log(p(x)). Can be unnormalized (just using gradient.)

    Returns:

    * the expectation of the Stein operator $\mathcal A [\text{fun}]$ wrt the empirical distribution of the particles xs:
    \[1/n \sum_i \mathcal A_p [\text{fun}](x_i) \]
    np.array of shape (d,) if transposed else shape (d, d)
    * if aux: per-particle drift and repulsion terms
    """
    out = vmap(
        stein_operator,
        (None, 0, None, None, None)
    )(fun, xs, logp, transposed, aux)

    if aux:
        steins, auxdata = out
        return np.mean(steins, axis=0), np.mean(auxdata, axis=0) # per-particle drift and repulsion, shape (2, d)
    else:
        steins = out
        return np.mean(steins, axis=0)

def phistar_i(xi, x, logp, kernel, aux=True):
    """
    Arguments:
    * xi: np.array of shape (d,), usually a row element of x
    * x: np.array of shape (n, d)
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y.

    Returns:
    * \phi^*(xi) estimated using the particles x (shape (d,))
    * auxdata consisting of [mean_drift, mean_repulsion] of shape (2, d)
    """
    if xi.ndim > 1:
        raise ValueError(f"Shape of xi must be (d,). Instead, received shape {xi.shape}")
    kx = lambda y: kernel(y, xi)
    return stein_expectation(kx, x, logp, aux=aux)

def get_phistar(kernel, logp, samples):
    def phistar(x):
        return phistar_i(x, samples, logp, kernel, aux=False)
    return phistar


def phistar(particles, leaders, logp, kernel):
    """
    O(nl) where n=#particles, l=#leaders
    Returns an np.array of shape (n, d) containing values of phi^*(x_i) for i in {1, ..., n}.

    Arguments:
    * particles: np.array of shape (n, d)
    * leaders: np.array of shape (l, d). Usually a subsample l < n of particles.
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y.

    Returns:
    * updates: np array of same shape as followers (n, d)
    * auxdata consisting of [mean_drift, mean_repulsion] of shape (n, 2, d)
    """
    return vmap(phistar_i, (0, None, None, None, None))(
        particles, leaders, logp, kernel, True)

def phistar_u(followers, leaders, logp, kernel):
    """
    O(l(l+m)) where m=#followers, l=#leaders
    Differences to phistar:
       follower updates are identical. Leader updates are computed in a 'leave-one-out'
    manner---that is, terms of the form k(x_i, x_i) are left out of the sum.

    Returns an np.array of shape (l+m, d) containing values of phi^*(x_i)
    for i in {1, ..., n}.

    Arguments:
    * followers: np.array of shape (m, d)
    * leaders: np.array of shape (l, d).
    * logp: callable
    * kernel: callable. Takes as arguments two vectors x and y of shape (d,)

    Returns:
    * per-particle updates: np array shape (l+m, d)
    * per-particle auxdata consisting of [mean_drift, mean_repulsion] shape (l+m, 2, d)
    """
    # Compute phi matrix of shape (l+m, m)
    # regular updates would then be np.sum(phi_matrix, axis=1) / n
    # instead, set leader subdiagonal of (l, l) submatrix to zero
    # then return row sum divided by l resp. (l-1).
    def f(x, y):
        """A^y_p k(x, y), evaluated inside the expectation wrt y
        x, y are both np.arrays of shape (d,), even when d=1.
        """
        kx = lambda y: kernel(x, y)
        return stein_operator(kx, y, logp, transposed=False, aux=True)
    fv  = vmap(f,  (None, 0))
    fvv = vmap(fv, (0, None))

    particles = np.concatenate([leaders, followers], axis=0)
    phi_matrix, auxdata = fvv(particles, leaders) # auxdata has shape (m+l, l, 2, d)
                                                  # phi_matrix shaped (m+l, l, d)
    l = leaders.shape[0]
    m = followers.shape[0]
    d = leaders.shape[1]
    trace_indices = [list(range(l))]*2
    phi_matrix = index_update(phi_matrix, trace_indices, 0)
    divisors = np.repeat(np.array([l-1, l]), np.array([l, m]))
    phi     = np.sum(phi_matrix, axis=1) / divisors.reshape((l+m, 1))
    auxdata = np.sum(auxdata,    axis=1) / divisors.reshape((l+m, 1, 1))
    return phi, auxdata

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

#@partial(jit, static_argnums=(1, 2, 3))
def ksd_squared_u(xs, logp, k, include_stddev=False):
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
    def h(x, y):
        """x, y: np.arrays of shape (d,)"""
        def inner(x):
            kx = lambda y_: k(x, y_)
            return stein_operator(kx, y, logp)
        return stein_operator(inner, x, logp, transposed=True)
    hv  = vmap(h,  (0, None))
    hvv = vmap(hv, (None, 0))
    ksd_matrix = hvv(xs, xs)
    n = xs.shape[0]
    diagonal_indices = [list(range(n))]*2
    ksd_matrix = index_update(ksd_matrix, diagonal_indices, 0)
    ksd_squared = np.sum(ksd_matrix) / (n * (n-1))
    if include_stddev:
        stddev = np.sqrt(np.clip(variance.var_ksd(ksd_matrix), a_min=1e-4))
        return ksd_squared, stddev
    else:
        return ksd_squared

#@partial(jit, static_argnums=(1,2))
def ksd_squared_v(xs, logp, k, dummy_arg1, dummy_arg2):
    """
    V-statistic for KSD^2. Computation in O(n^2)
    Arguments:
    * xs: np.array of shape (n, d)
    * logp: callable
    * k: callable, computes scalar-valued kernel k(x, y) given two input arguments.

    Returns:
    The square of the stein discrepancy KSD(q, p).
    KSD is approximated as $1 / n^2 \sum_{i, j} g(x_i, x_j)$, where the x are iid distributed as q
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

    return np.sum(ksd_matrix) / n**2

#@partial(jit, static_argnums=(1,2,3))
def ksd_squared_l(samples, logp, k, return_variance=False):
    """
    O(n) time estimator for the KSD.
    Arguments:
    * samples: np.array of shape (n, d)
    * logp: callable
    * k: callable, computes scalar-valued kernel k(x, y) given two input arguments of shape (d,).

    Returns:
    * The square of the stein discrepancy KSD(q, p).
    KSD is approximated as $\sum_i g(x_i, y_i)$, where the x and y are iid distributed as q
    * The approximate variance of h(X, Y)
    """
    try:
        xs, ys = samples.split(2)
    except ValueError: # uneven split
        xs, ys = samples[:-1].split(2)

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


def stein_discrepancy(xs, logp, f, aux=False):
    """Return estimated stein discrepancy using
    witness function f"""
    return stein_expectation(f, xs, logp, transposed=True, aux=aux)

def h(x, y, kernel, logp):
    k=kernel
    def h2(x_, y_): return np.inner(grad(logp)(y_), grad(k, argnums=0)(x_, y_))
    def d_xk(x_, y_): return grad(k, argnums=0)(x_, y_)
    out = np.inner(grad(logp)(x), grad(logp)(y)) * k(x,y) +\
            h2(x, y) + h2(y, x) +\
            np.trace(jacfwd(d_xk, argnums=1)(x, y))
    return out

def g(x, y, kernel, logp):
    """x, y: np.arrays of shape (d,)"""
    k=kernel
    def inner(x):
        kx = lambda y_: k(x, y_)
        return stein_operator(kx, y, logp)
    return stein_operator(inner, x, logp, transposed=True)

def globally_maximal_stein_discrepancy(proposal, target, lambda_reg=1):
    """Returns Stein discrepancy E_{x \sim q}[\mathcal A_p f(x)] using
    witness function f(x) = (\nabla \log q(x) - \nabla \log p(x)) / (2*lambda).

    This f is the optimal (KSD-maximizing) function among all functions with
    (figure out exact criterion by looking at lagrange multipliers)"""
    def optimal_witness(x): # gradient of KL(x || p)
        return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x)

    def stein_op_true(x):
        return np.inner(optimal_witness(x), optimal_witness(x))

    @jit
    def stein_discrepancy(samples):
        return np.mean(vmap(stein_op_true)(samples))
    return stein_discrepancy(proposal.sample(1000))


# tests
def test_h_successful():
    target = distributions.Gaussian(0, 5)
    source = distributions.Gaussian(3, 1)
    k = kernels.get_rbf_kernel_logscaled(logh=0)
    logp = target.logpdf
    x, y = source.sample(2)
    return g(x, y, k, logp) == h(x, y, k, logp)

assert test_h_successful()
