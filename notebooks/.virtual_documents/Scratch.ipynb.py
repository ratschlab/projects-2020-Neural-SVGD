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


string_f1 = "kxkjdf![img](/home/lauro/obsidian/Pasted image 14.png)xdkfjdlk\n"
string_f2 = "![[Pasted image 14.png]]sdlfsdfsdfj\n"


def f1_to_f2(string):
    ind_begin = string.rindex("/") # last occurence of /
    ind_end = string.rindex(")") # last occurence of )
    name = string[ind_begin+1:ind_end]
    return f"![[{name}]]"

def f2_to_f1(string):
    ind_begin = string.index("!")
    ind_end = string.rindex("]")
    name = string[ind_begin+3:ind_end-1]
    return f"![img](/home/lauro/obsidian/{name})"

def line_is_f1_img(line):
    return line.startswith("![img]") and line.endswith(".png)\n")

def line_is_f2_img(line):
    return line.startswith("![[") and line.endswith(".png]]\n")


def convert_file(filename, direction="f1_to_f2"):
    if direction == "f1_to_f2":
        convert = f1_to_f2
        is_img = line_is_f1_img
    elif direction == "f2_to_f1":
        convert = f2_to_f1
        is_img = line_is_f2_img
    else:
        raise ValueError()
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if is_img(line):
                line_converted = convert(line)
                lines[i] = line_converted + "\n"
                print(f"changed line: {line}")
    with open(filename, "w") as f:
        f.writelines(lines)


# filename = "/home/lauro/testfile"
# convert_file(filename, direction="f1_to_f2")


filename = "/home/lauro/obsidian/Master thesis/Updates/Update September 8.md"


convert_file(filename, direction="f1_to_f2")
