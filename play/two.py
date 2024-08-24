# %% two.py
#     play two (autograd)
# by: Noah Syrkis

# %% Imports ############################################################
from jax import random, grad, Array
import jax.numpy as jnp
from typing import Callable, Dict, List
import chex


# %% Partial potential solution
@chex.dataclass
class Value:
    value: Array
    parents: List["Value"] | None = None
    grad_fn: Callable | None = None


def add(x: Value, y: Value) -> Value:
    """Add two values."""
    raise NotImplementedError


def mul(x: Value, y: Value) -> Value:
    """Multiply two values."""
    raise NotImplementedError


def backward(x: Value, grad: Array) -> Dict[Value, Array]:
    """Backward pass."""
    raise NotImplementedError


def update(x: Value, grad: Array) -> Value:
    """Apply the gradient to the value."""
    raise NotImplementedError
