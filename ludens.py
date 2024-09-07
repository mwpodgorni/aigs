# %% Imports
import jax.numpy as jnp  # <- the core of jax are jnp.ndarrays
from jax import grad, random, nn, tree, jit
from tqdm import tqdm  # <- for progress bars
import tensorflow_datasets as tfds  # <- for data
import seaborn as sns
import matplotlib.pyplot as plt

# %% Load the data
mnist = tfds.load("mnist", split="train")  # <- load the mnist dataset
xs, ys = [], []  # <- create empty lists to store the data
for sample in tfds.as_numpy(mnist):  # <- iterate over the dataset
    x = sample["image"]  # <- extract the image
    y = sample["label"]  # <- extract the label
    xs.append(x)  # <- append the image to the list
    ys.append(y)  # <- append the label to the list
x = jnp.array(xs).reshape(60_000, 28 * 28) / 255.0  # <- convert the list to a jnp.ndarray
y = jnp.eye(10)[jnp.array(ys)]
batch_size = 32
x, y = x.reshape(-1, batch_size, 28 * 28), y.reshape(-1, batch_size, 10)

# %%
def model(x, params):
    w1, w2, b1, b2 = params
    z = nn.relu(x @ w1 + b1) @ w2 + b2
    y_hat = nn.softmax(z, axis=1)
    return y_hat

def loss_fn(param, x, y):
    y_hat = model(x, param)
    return ((y - y_hat) ** 2).mean()

grad_fn = jit(grad(loss_fn))



keys = random.split(random.PRNGKey(0), 4)
w1 = random.normal(keys[0], (28 * 28, 32)) * 0.01
w2 = random.normal(keys[1], (32, 10)) * 0.01
b1 = random.normal(keys[2], (32,)) * 0.01
b2 = random.normal(keys[3], (10,)) * 0.01
params = (w1, w2, b1, b2)

# %%

def update(p, g):
    return p - 0.01 * g

pbar = tqdm(range(20))
for epoch in pbar:
    for x_batch, y_batch in zip(x, y):
        grads = grad_fn(params, x_batch, y_batch)
        params = tree.map(update, params, grads)
        loss = loss_fn(params, x_batch, y_batch)
        pbar.set_description(f"Loss: {loss:.2f}")  # type: ignore

# %%
y_hat = model(x, params)
(y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
