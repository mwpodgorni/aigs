# %% mlp.py
#    this script showcases a simple linear regression model
# by: Noah Syrkis

# %% imports
import jax.numpy as jnp
import jax
from jax import random, tree, grad
import chex


# %% Initialization
def init_linear(key: chex.PRNGKey, in_features: int, out_features: int):
    w = random.normal(key, (in_features, out_features))
    b = random.normal(key, (out_features,))
    return w, b


# %% Loss function
def loss_fn(params, x, y):
    return jnp.mean((y - model(params, x)) ** 2)


# %% Forward pass
def model(params, x):
    (w1, b1), (w2, b2) = params
    z = jax.nn.relu(jnp.dot(x, w1) + b1)
    z = jnp.dot(z, w2) + b2
    return z


# %% Create silly data
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = jnp.array([[1.5], [2.5], [3.5]])

# %% Initialize parameters
rng = random.PRNGKey(0)
rng, key1, key2 = random.split(rng, 3)

layer_1 = init_linear(key1, 2, 4)
layer_2 = init_linear(key2, 4, 1)

params = (layer_1, layer_2)
lr = 0.001

# %% Training loop
for i in range(10):
    grads = grad(loss_fn)(params, x, y)
    params = tree.map(lambda p, g: p - lr * g, params, grads)

    loss = loss_fn(params, x, y)
    print(f"Loss: {loss}")
