import jax.numpy as np
from jax import grad, jit, vmap, random, value_and_grad
from jax import lax
from jax.ops import index, index_add, index_update
from jax.experimental import optimizers

import time
from tqdm import tqdm
import warnings
from functools import partial

import utils
import metrics
import stein
import kernels

verbose_jit = utils.verbose_jit

# config
kernel = kernels.ard
dist = metrics.Gaussian(0, 1) # target dist
logp = dist.logpdf
n = 100 # nr particles

lr_svgd = 1
lr_ksd = 1e-2

n_svgd_steps = 300
ksd_steps_per_svgd_step = 1


class Optimizer():
    def __init__(self, opt_init, opt_update, get_params):
        """opt_init, opt_update, get_params are the three functions optained from a stax.optimizer call."""
        self.init = verbose_jit(opt_init)
        self.update = verbose_jit(opt_update)
        self.get_params = verbose_jit(get_params)

        self.state = None


@partial(verbose_jit, static_argnums=1)
def init_svgd(key, particle_shape):
    return random.normal(key, particle_shape) * 2 - 3

@partial(verbose_jit, static_argnums=1)
def init_kernel(key, kernel_param_shape):
    return random.normal(key, kernel_param_shape)

class SVGD():
    def __init__(self, target, n_particles, kernel, lr_svgd, optimizer_svgd):
        self.target = target
        self.n_particles = n_particles
        self.kernel = kernel
        self.lr_svgd = lr_svgd
        self.lr_ksd = lr_ksd
        self.opt = optimizer_svgd

        self.n = self.n_particles
        self.particle_shape = (self.n, self.target.d)
        self.kernel_param_shape = () # 1d for now

    @partial(verbose_jit, static_argnums=0)
    def phistar(self, particles, kernel_params):
        return stein.phistar(particles, self.target.logpdf, self.kernel(kernel_params))

    @partial(verbose_jit, static_argnums=0)
    def ksd_squared(self, kernel_params, particles):
        return stein.ksd_squared(particles, self.target.logpdf, self.kernel(kernel_params))

    def train_kernel(self, key, n_svgd_steps, ksd_steps_per_svgd_step, lr_ksd, opt_ksd):
        """Train the kernel parameters during one SVGD run.
        Arguments
        opt_ksd : instance of class Optimizer.

        Returns
        Updated kernel parameters.
        """

        log = {"particles": [],
                "kernel_param": [],
                "ksd": []}

        particles = init_svgd(key, self.particle_shape)
        kernel_params = init_kernel(key, self.kernel_param_shape)
        self.opt.state = self.opt.init(particles) # TODO: bad solution, since state is now stored in SVGD instance. So need to clear later to prevent mixups of states between separate training or sampling runs.
        # On second thought: maybe that's fine.
        # hell, it's even good: now samples stored in instance.
        # hm tho maybe that isn't good after all
        opt_ksd.state = opt_ksd.init(kernel_params)

        for i in range(n_svgd_steps):
            particles = self.opt.get_params(self.opt.state)
            log["particles"].append(particles)

            # update kernel_params:
            inner_updates = []
            ksds = []
            for j in range(ksd_steps_per_svgd_step):
                kernel_params = opt_ksd.get_params(opt_ksd.state)
                inner_updates.append(kernel_params)
                ksd, gk = value_and_grad(self.ksd_squared)(kernel_params, particles)
                opt_ksd.state = opt_ksd.update(j + i*ksd_steps_per_svgd_step, gk, opt_ksd.state)
                ksds.append(ksd)

            log["ksd"].append(ksds)
            log["kernel_param"].append(inner_updates)


            # update particles:
            kernel_params = opt_ksd.get_params(opt_ksd.state)
            gp = -self.phistar(particles, kernel_params)
            self.opt.state = self.opt.update(i, gp, self.opt.state)

        return kernel_params, log

    # @partial(verbose_jit, static_argnums=(0, 3))
    def sample(self, key, kernel_params, n_svgd_steps):
        """Sample from the (approximation of the) target distribution using the current kernel parameters."""
        particles = init_svgd(key, self.particle_shape)
        self.opt.state = self.opt.init(particles)

        for i in range(n_svgd_steps):
            particles = self.opt.get_params(self.opt.state)
            gp = -self.phistar(particles, kernel_params)
            self.opt.state = self.opt.update(i, gp, self.opt.state)

        return self.opt.get_params(self.opt.state)


if __name__ == "main":
    # config
    kernel = kernels.ard
    target = metrics.Gaussian(0, 10) # target dist
    n = 500 # nr particles

    lr_svgd = 1
    lr_ksd = 1e-2

    n_svgd_steps = 300
    ksd_steps_per_svgd_step = 1

    opt_svgd = train.Optimizer(*optimizers.sgd(step_size=lr_svgd))
    opt_ksd  = train.Optimizer(*optimizers.momentum(step_size=lr_ksd, mass=0.9))

    today = time.strftime("%Y-%m-%d")
    logfile = f"results/{today}"
    try:
        os.makedirs('results')
    except:
        pass

    try:
        os.makedirs(logfile)
    except:
        pass


    # run SVGD
    print("Learning kernel parameters...")
    s = train.SVGD(target, n, kernel, lr_svgd, opt_svgd)
    params, log = s.train_kernel(rkey, n_svgd_steps, ksd_steps_per_svgd_step, lr_ksd, opt_ksd)

#    # write to logs
#    print(f"Writing results to {logfile}.")
#    with open(logfile, "w") as f:
#        f.write()
