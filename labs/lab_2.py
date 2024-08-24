# %% lab_2.py
#     play two (autograd) and neural networks
# by: Noah Syrkis

#########################################################################
# %% Imports (add as needed) ############################################
from jax import grad
import jax.numpy as jnp
from jaxtyping import Array
import chex
from typing import Callable, Dict, List
import equinox as eqx
import os

import tensorflow_datasets as tfds  # <- for data

import seaborn as sns  # <- for plotting
import matplotlib.pyplot as plt  # <- for plotting


#########################################################################
# %% Neural Networks ####################################################
# Implement a simple neural network using jnp, grad and random.
# Use the MNIST dataset from tensorflow_datasets.

# %% Data
mnist = tfds.load("mnist", split="train")
x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)])
y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

# %% Params (define model params in a dict (or list, of chex.dataclass))

# %% Model (define model as a function that takes params and x_data)

# %% Loss (define loss as a function that takes params, x_data, y_data)

# %% Train loop

# %% play with JIT and vmap to speed up training and simplify code


#########################################################################
# %% Equinox (optional) #################################################
# Do the same in Equinox (high-level JAX library)


#########################################################################
# %% Autograd (optional) ################################################
# Implement a simple autograd system using chex and jax (no grad)
# Structure could be something like this:


@chex.dataclass
class Value:
    value: Array
    parents: List["Value"] | None = None
    gradient_fn: Callable | None = None


def add(x: Value, y: Value) -> Value:
    """Add two values."""
    raise NotImplementedError


def mul(x: Value, y: Value) -> Value:
    """Multiply two values."""
    raise NotImplementedError


def backward(x: Value, gradient: Array) -> Dict[Value, Array]:
    """Backward pass."""
    raise NotImplementedError


def update(x: Value, gradient: Array) -> Value:
    """Apply the gradient to the value."""
    raise NotImplementedError
