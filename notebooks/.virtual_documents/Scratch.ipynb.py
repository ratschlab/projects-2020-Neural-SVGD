import sys
import copy
import os
sys.path.append("/home/lauro/code/msc-thesis/svgd/kernel_learning")
import json
import collections
import itertools
from functools import partial
import importlib

import numpy as onp
from jax.config import config
# config.update("jax_log_compiles", True)
# config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import grad, jit, vmap, random, lax, jacfwd, value_and_grad
from jax import lax
from jax.ops import index_update, index
import matplotlib.pyplot as plt

import numpy as onp
import jax
import pandas as pd
import haiku as hk
import ot

import config

import utils
import metrics
import time
import plot
import stein
import kernels
import distributions
import nets
import kernel_learning

from jax.experimental import optimizers

key = random.PRNGKey(0)

from jax.scipy.stats import norm


target = distributions.GaussianMixture([-5, 5], [1, 9], [1, 1])
proposal = distributions.GaussianMixture([-6, 0, 6], [9, 1, 1], [1,1,1])
plot.plot_fun(target.pdf, lims=(-10, 10), label="target")
plot.plot_fun(proposal.pdf, lims=(-10, 10))
plt.legend()


target = distributions.GaussianMixture([-5, 5], [1, 9], [1, 1])
proposal = distributions.Gaussian(0, 3)
plot.plot_fun(target.pdf, lims=(-10, 10), label="target")
plot.plot_fun(proposal.pdf, lims=(-10, 10))
plt.legend()


target = distributions.Banana([0, 0], [4, 1])
proposal = distributions.Gaussian([0, 0], [4, 4])
plot.plot_fun_2d(target.pdf, lims=(-10, 10), label="target")
plot.plot_fun_2d(proposal.pdf, lims=(-10, 10))
plt.legend()


plot.scatter(target.sample(1000))


target = distributions.Funnel(2)
proposal = distributions.Gaussian([0,0], 9)
plot.plot_fun_2d(target.pdf, lims=(-10, 10), label="target")
plot.plot_fun_2d(proposal.pdf, lims=(-10, 10))
plt.legend()


target = distributions.Ring(10, .1)
proposal = distributions.Gaussian([0,0], 9)
plot.plot_fun_2d(target.pdf, lims=(-10, 10))
# plot.plot_fun_2d(proposal.pdf, lims=(-10, 10))
plt.legend()


plot.scatter(target.sample(1000))
plot.scatter(proposal.sample(1000))


target = distributions.Ring(10, .1)
proposal = distributions.Ring(15, 0.1)
plot.plot_fun_2d(target.pdf, lims=(-15, 15))
plot.plot_fun_2d(proposal.pdf, lims=(-20, 20))
plt.legend()


plot.scatter(target.sample(1000))
plot.scatter(proposal.sample(1000))


sldfkskj


x = np.array([1,2])
y = np.array([-1,3])


norm.pdf(x, loc=y)


x = np.array(0)


plot.plot_fun(lambda y: kernels.get_tophat_kernel(4, normalize=True)(y, x))


sldkfj


particles = np.arange(10)
group_idx = random.permutation(key, np.arange(len(particles))).split(2)
leaders, followers = [particles[idx] for idx in group_idx]


mixed_particles = np.concatenate([leaders, followers])


particles_again = index_update(mixed_particles, np.concatenate(group_idx), mixed_particles)


particles_again


d = distributions.Gaussian(0, 1)
target = distributions.Gaussian(0, 10)
kernel = kernels.get_rbf_kernel(1)

followers = d.sample(200)
leaders = d.sample(200)
particles = np.concatenate([leaders, followers])


out, aux = stein.phistar(particles, leaders, target.logpdf, kernel)


out_u, aux_u = stein.phistar_u(followers, leaders, target.logpdf, kernel)


np.max(out - out_u)


np.max(aux - aux_u)


asjdh


def vmv_dot(vec_a, matrix, vec_b):
    """
    Returns x^T A x, the vector-matrix-vector dot product
    """
    return np.einsum("i,ij,j->", vec_a, matrix, vec_b)


x = np.array([1,2])
m = np.eye(2)


vmv_dot(x, m, x)


target = distributions.Funnel(2)
proposal = distributions.Uniform([[-1, 1], [-6, 6]])
kernel = kernels.get_rbf_kernel(1)
kernel = kernels.get_funnel_kernel(1)


sample = proposal.sample(400)
stein.ksd_squared_u(sample, target.logpdf, kernel)


x = np.asarray([1])
scale=np.asarray(2)
jax.scipy.stats.uniform.pdf(x, loc=x, scale=scale)


plot.plot_fun(lambda x: jax.scipy.stats.uniform.pdf(x, scale=scale), (-5, 5))


target = distributions.Gaussian(0, 0.1)
proposal = distributions.Gaussian(0, 5)
n_particles = 200
n_iter = 5
n_iter_kernel = 20
svgd_lr = 0.05
kernel_lr = 0.1
sizes = [1]

key, subkey, svgd_key = random.split(key, 3)
learner = kernel_learning.AdversarialSVGD(subkey, target, proposal, sizes, svgd_lr=svgd_lr, kernel_lr=kernel_lr, svgd_key=svgd_key, n_particles=n_particles)

s = kernel_learning.SVGD(
    svgd_key, target, proposal,
    n_particles=n_particles,
    learning_rate=svgd_lr, 
    get_kernel=lambda par: kernels.get_rbf_kernel(1.),
)


np.mean(s.get_params())


np.mean(learner.svgd.get_params())


slkdfj


prop = distributions.Gaussian(0, 1)
target = distributions.Gaussian(0, 15)
samples = prop.sample(100)
kernel = kernels.get_rbf_kernel(1)

eps = 1.
# @jit
def T(x):
    return x + eps * stein.phistar_i(x, samples, target.logpdf, kernel, False)

x = np.asarray([1.])


jacfwd(T)(x)


np.linalg.det(jacfwd(T)(x))


prop = distributions.Gaussian(0, 1)
samples = prop.sample(5)
t = lambda x: x*2
loglike = metrics.pushforward_loglikelihood(t, vmap(prop.logpdf)(samples), samples)

print(samples)
print(np.exp(loglike))
print(np.exp(vmap(prop.logpdf)(samples))/2)


def sample_gauss(key, shape, mean, std):
    return random.normal(key, shape) * std + mean




target = distributions.Gaussian(0, 15)
learner = kernel_learning.KernelLearner(key, target, [1], kernels.get_rbf_kernel(1), .1, 0)


for _ in range(50):
    samples = 


import kernel_learning
target = distributions.Gaussian(0, 10)
proposal = distributions.Gaussian(0, 1)
key, subkey = random.split(key)
s = kernel_learning.SVGD(subkey, target, proposal, 200)

key, subkey = random.split(key)
s.flow()

plt.plot(s.rundata["leader_variance"])


s.rundata.keys()


ssdf;dfl


def dirac_kernel(x, y):
    return np.where(x==y, 1., 0.)



dirac_kernel(1, 1)


def add(x, y): return x + y
madd = vmap(vmap(add))

n=1000
x = random.normal(key, (n,n))


time = get_ipython().run_line_magic("timeit", " -o madd(x,x).block_until_ready()")


regular_time = get_ipython().run_line_magic("timeit", " -o add(x,x).block_until_ready()")


jadd = jit(add)
jmadd = jit(madd)


time = get_ipython().run_line_magic("timeit", " -o jmadd(x,x).block_until_ready()")


regular_time = get_ipython().run_line_magic("timeit", " -o jadd(x,x).block_until_ready()")


from kernels import defunnelize
f = distributions.Funnel(2)
s = f.sample(100)


def defunnelize(z):
    """Inverse of funnelize."""
    *x, y = np.asarray(z)
    print(x)
    print(y)
    return x, y



x, y = defunnelize(s[0])


x*y


vmap(defunnelize)(s)


fg = distributions.FunnelizedGaussian([0,0], 1)


s = random.normal(key, (100, 2)) *3


(vmap(fg.pdf)(s) - vmap(fg._pdf)(s)).max()


s = fg.sample(1000)
s.shape


fig, ax = plt.subplots(figsize=[8, 8])
plot.scatter(s, ax=ax, marker=".")
ax.set_xlim((-20, 20))
ax.set_ylim((-10, 10))





sdkf


class Base():
    def __init__(self):
        self.c = 2
    
    def method(self, x):
        return self.c*x
    
class Sub(Base):
    def __init__(self):
        super().__init__()
        self.c = 10
        
    def method(self, x):
        return self.c*super(Sub, self).method(x)


test = Sub()
test.method(1)


conf_mlp = copy.deepcopy(config.config)

# conf_mlp["svgd"]["target_args"] = [[0, 5], [1, 3]]
conf_mlp["svgd"]["target"] = "Gaussian Mixture"
conf_mlp["svgd"]["target_args"] = metrics.bent_args

conf_mlp["svgd"]["n_particles"] = 6000
conf_mlp["svgd"]["n_subsamples"] = 100
conf_mlp["svgd"]["lam"] = 1
conf_mlp["svgd"]["encoder_layers"] = [8, 8, 8, 2]
conf_mlp["svgd"]["decoder_layers"] = [8, 8, 4, 1]

conf_mlp["train_kernel"]["ksd_steps"] = 2
conf_mlp["train_kernel"]["svgd_steps"] = 1
conf_mlp["train_kernel"]["n_iter"] = 80 #config.config["train_kernel"]["n_iter"] // conf_mlp["train_kernel"]["svgd_steps"]
conf_mlp["train_kernel"]["optimizer_ksd_args"] = [0.03]
conf_mlp["train_kernel"]["lamda_reg"] = 1e-2


s = SVGD(**config.get_svgd_args(conf_mlp))


params, rundata = s.train_kernel(key, **config.get_train_args(conf_mlp))


plt.plot(rundata["ksd_before_kernel_update_val"])
plt.yscale("log")


d = distributions.GaussianMixture(*conf_mlp["svgd"]["target_args"])
k = kernels.get_rbf_kernel_logscaled(logh=0)
# k = lambda x, y: np.dot(x, y)
# k = s.get_kernel_fn(params)


eps = 1e-6
n = 100
x = d.sample(n)
gram = vmap(vmap(k, (0, None)), (None, 0))(x, x)
gram_reg = gram + eps * np.identity(n)
print("gram PD:", utils.is_pd(gram))
print("regularized gram PD:", utils.is_pd(gram_reg))


vals, vecs = np.linalg.eig(gram)
vals = np.asarray(vals, dtype=np.float32)

vals_reg, _ = np.linalg.eig(gram_reg)
vals_reg = np.asarray(vals_reg, dtype=np.float32)


np.min(vals)


np.min(vals_reg)


def f(key): return random.normal(key)
def g(key): return random.normal(key) + 30

mix = utils.mixture([f, g], [0.2, 0.8])


x = []
for key in random.split(key, 1000):
    x.append(mix(key))


_ = plt.hist(x, bins=100)


key, subkey = random.split(key)
random.randint(subkey, shape=(1,), minval=0, maxval=2)


l = np.asarray((1, 2, 1.5, 3, 3.3, 3.8))
l = onp.concatenate([-l, [0], l])
means = list(zip(l, (l**2)**0.8))
variances = [[1,1]]*len(means)
weights = [1]*len(means)
target = distributions.GaussianMixture(means, variances, weights)


plot.plot_pdf(target.pdf, (-10, 10), "contour")


plot.bivariate_hist(target.sample(100))


target = distributions.Gaussian(0, 1)
source = distributions.Gaussian(0, 1)
kernel = kernels.get_rbf_kernel_logscaled(logh=0)


L = 10 # linear time estimator uses L2n samples --> memory L2n, computation Ln, Var(KSD_L) = 1/Ln Var(h(X, Y))
      # where n = nr of samples for U-estimator


def sample_ksd(n, source, target, mode="l"):
    if mode=="u":
        samples = source.sample(n)
        return stein.ksd_squared_u(samples, target.logpdf, kernel)
    elif mode=="l":
        samples = source.sample(2*n).split(2)
        return stein.ksd_squared_l(*samples, target.logpdf, kernel, False)
    else:
        raise ValueError("mode must be 'u' or 'l'.")

# @partial(jit, static_argnums=range(5))
def sample_ksd_variance(n, m, source, target, mode="l"):
    return np.var(np.asarray([sample_ksd(n, source, target, mode) for _ in range(m)]))


m = 100
vars_u = []
vars_l = []
ngrid = 2**onp.arange(5, 10)

for n in ngrid:
    print(n)
    vars_u.append(sample_ksd_variance(n, m, source, target, "u"))
    vars_l.append(sample_ksd_variance(L*n, m, source, target, "l"))


vars_u, vars_l = [np.asarray(x) for x in (vars_u, vars_l)] 

if not np.all(vars_l / vars_u < 1):
    print("Variance is too high!!")


plt.plot(ngrid, vars_l / vars_u, ".")
plt.xscale("log")


vars_l


vars_u


m = 1000000
samples = source.sample(m).split(2)
hvar = stein.h_var(*samples, target.logpdf, kernel)
hvar

plt.plot(ngrid, ngrid * L * vars_l, label="V_L in practice")
plt.hlines(hvar, xmin=0, xmax=ngrid[-1], label="V_L in theory")
# plt.yscale("log")
plt.xscale("log")
plt.legend()


plt.plot(ngrid, ngrid * vars_u, ".")
plt.xscale("log")


def means(xs):
    compute = [np.mean(xs) for _ in range(100)], [np.mean(xs*2)/2 for _ in range(100)]
    return np.asarray(compute)
jitmeans = jit(means)
@jit
def single_mean(xs): return np.mean(xs)


key, subkey = random.split(key)
x = random.normal(subkey)

get_ipython().run_line_magic("timeit", " -o test(x).block_until_ready()")
get_ipython().run_line_magic("timeit", " -o jittest(x).block_until_ready()")
print()


slkfj


def vmapped_id(x, n):
    xs = np.ones(shape=(n, n, n))*x
    idv = vmap(lambda x: x)
    idvv = vmap(idv)
    idvvv = vmap(idvv) # now id takes argument of shape (n, n, n)
    return np.mean(idvvv(xs))

jitid = jit(vmapped_id, static_argnums=1)

x = random.normal(key)

get_ipython().run_line_magic("timeit", " vmapped_id(x, 10).block_until_ready()")
get_ipython().run_line_magic("timeit", " jitid(x, 10).block_until_ready()")


m = 10000
# vmapped_id(x, m).block_until_ready()   # runtime error (OOM) 3.6 TB
# jitid(x, m).block_until_ready() # also a runtime error (OOM) 3.6 TB





sldfj


n = 10**2
key, subkey = random.split(key)
x = random.normal(subkey, (n,))


normalization_factor = np.sum(norm.pdf(x))
weights = norm.pdf(x) / normalization_factor


np.mean(x**2 * weights)


np.mean(x**2)





n = 10**6
key, subkey = random.split(key)
x = random.normal(subkey, (n,))

np.any(jax.scipy.stats.norm.pdf(x) == 0)
