# %% lab_2.py
#     play two (autograd) and neural networks
# by: Noah Syrkis

#########################################################################
# %% Imports (add as needed) ############################################
from jax import grad, random, nn, tree
import jax.numpy as jnp
from jaxtyping import Array
import chex
import equinox as eqx
import optax

from typing import Callable, Dict, List
from tqdm import tqdm
import os

import tensorflow_datasets as tfds  # <- for data

import seaborn as sns  # <- for plotting
import matplotlib.pyplot as plt  # <- for plotting


#########################################################################
# %% Neural Networks ####################################################
# Implement a simple neural network using jnp, grad and random.
# Use the MNIST dataset from tensorflow_datasets.

# %% Data
# mnist = tfds.load("mnist", split="train")
# x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1, 28 * 28)
# y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

# %%
x_data.shape, y_data.shape

# %% Plot
hidden = 32

# %% Params (define model params in a dict (or list, of chex.dataclass))
rng = random.PRNGKey(0)
w1 = random.normal(rng, (28 * 28, hidden)) * 0.01
b1 = random.normal(rng, (hidden,)) * 0.01
w2 = random.normal(rng, (hidden, 10)) * 0.01
b2 = random.normal(rng, (10,)) * 0.01


# %% Model (define model as a function that takes params and x_data)
def model(params, x_data):
    w1, b1, w2, b2 = params
    z = x_data @ w1 + b1
    z = nn.relu(z)
    z = z @ w2 + b2
    z = nn.softmax(z)
    return z


# %% Loss (define loss as a function that takes params, x_data, y_data)
def loss_fn(params, x_data, y_data):  # y: n
    y_hat = model(params, x_data)
    y_data = jnp.eye(10)[y_data]  # <- one-hot encode  # y: n x 10
    loss = jnp.mean((y_hat - y_data) ** 2)
    return loss


# %% Train loop
params = [w1, b1, w2, b2]
for i in tqdm(range(100)):
    grads = grad(loss_fn)(params, x_data, y_data)
    params = [params[i] - 0.01 * grads[i] for i in range(len(params))]
    # params = tree.map(lambda p, g: p - 0.01 * g, params, grads)  # <- same as above

    # params -= gradient_of_loss(params)
# %%
y_hat = model(params, x_data)
(y_hat.argmax(axis=1) == y_data).astype(jnp.int32).mean()
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
