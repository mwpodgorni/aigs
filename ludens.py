# %% Exercise 1
import jax.numpy as jnp  # JAX's version of numpy
from jax import random, tree, grad, nn  # JAX's utilities
import chex  # library for checking JAX types (nice for mental health)
from typing import List  # for type hints

# %% Types for exercise 2 and 3
Number = float | int
Vector = List[Number]
Matrix = List[Vector]


## Exercise 2 and 3
def dot(v: Vector, w: Vector) -> Number:
    raise ValueError("This function is not implemented")
    # return sum(v_i * w_i for v_i, w_i in zip(v, ...


def matmul(A: Matrix, B: Matrix) -> Matrix:
    raise ValueError("This function is not implemented")
    # return [[dot(row, col) for col in zip(*B)] for ...


# %% Exercise 4
def init_linear(key: chex.PRNGKey, in_features: int, out_features: int):
    w = random.normal(key, (in_features, out_features))
    b = random.normal(key, (out_features,))
    return w, b


def loss_fn(params, x, y):
    return jnp.mean((y - model(params, x)) ** 2)


def model(params, x):
    (w1, b1), (w2, b2) = params
    z = nn.relu(jnp.dot(x, w1) + b1)
    z = jnp.dot(z, w2) + b2
    return z


x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([[1.5], [2.5], [3.5]])

rng = random.PRNGKey(0)
rng, key1, key2 = random.split(rng, 3)

layer_1 = init_linear(key1, 2, 4)
layer_2 = init_linear(key2, 4, 1)

params = (layer_1, layer_2)
lr = 0.001

for i in range(10):
    grads = grad(loss_fn)(params, x, y)
    params = tree.map(lambda p, g: p - lr * g, params, grads)

    loss = loss_fn(params, x, y)
    print(f"Loss: {loss}")
