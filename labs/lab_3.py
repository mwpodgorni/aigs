# %% lab_4.py
#    deep learning with JAX
# by: Noah Syrkis

# %% Imports ############################################################
from jax import grad, lax, random, nn, tree, jit
import jax.numpy as jnp
from tqdm import tqdm
from jaxtyping import Array
import chex
from typing import Callable, Dict, List
import sklearn.datasets as skd
import tensorflow_datasets as tfds
import seaborn as sns
import matplotlib.pyplot as plt

# %%
mnist = tfds.load("mnist", split="train")
x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1, 1, 28, 28).astype(jnp.float32)
y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

# %% Convolutional neural networks ######################################
# Make a CNN classifying faces using JAX or Equinox.
# Compare the performance of your model to a simple neural network.
# 1. MAKE AND INIT PARAMS FUNCTION THAT RETURNS A DICTIONARY WITH KERNELS AND MLP PARAMS
# 2. MAKE AN APPLY FUNCTION THAT TAKES IN THE PARAMS AND INPUTS AND RETURNS THE OUTPUT
# NOTE: DO BOTH THE CONV STUFF, AND THE FULLY CONNECTED STUFF. REMEMBER ACTIVATION FUNCTIONS.
# 3. AT THIS POINT YOU HAVE AN OUTPUT "Y_HAT" FROM THE MODEL. USE GRADS TO OPTIMIZE WEIGHTS.
def conv(input, kernel, stride=1):
    return lax.conv(input, kernel, (stride, stride), padding='SAME')

def init_fn(rng, cfg=None):
    rng, *keys = random.split(rng, 10)
    kernel_1 = random.normal(keys[0], (3, 1, 3, 3))
    kernel_2 = random.normal(keys[1], (2, 3, 3, 3))
    w_1 = random.normal(keys[2], (1568, 128))
    b_1 = random.normal(keys[3], (128,))
    w_2 = random.normal(keys[4], (128, 10))
    b_2 = random.normal(keys[5], (10,))
    params = dict(kernel1=kernel_1, kernel2=kernel_2, w1=w_1, b1=b_1, w2=w_2, b2=b_2)
    return params

def apply_fn(params, x):
    z = conv(x, params['kernel1'])
    z = jnp.tanh(z)
    z = conv(z, params['kernel2'])
    z = z.reshape(60000, -1)
    z = z @ params['w1'] + params['b1']
    z = jnp.tanh(z)
    z = z @ params['w2'] + params['b2']
    return nn.softmax(z)

def loss_fn(params, x_data, y_data):  # y: n
    y_hat = apply_fn(params, x_data)
    y_data = jnp.eye(10)[y_data]  # <- one-hot encode  # y: n x 10
    loss = jnp.mean((y_hat - y_data) ** 2)
    return loss




# %%
rng = random.PRNGKey(0)
params = init_fn(rng)

grad_fn = jit(grad(loss_fn))
for i in tqdm(range(100)):
    grads = grad_fn(params, x_data, y_data)
    params = tree.map(lambda p, g: p - 0.01 * g, params, grads)  # <- same as above
#
# %%
# %%
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
axes[0].imshow(x_data[0, 0], cmap='gray')
axes[1].imshow(output[0, 0], cmap='gray')
axes[2].imshow(output[0, 1], cmap='gray')
axes[3].imshow(output[0, 2], cmap='gray')
axes[4].imshow(output[0, 3], cmap='gray')




# %% Autoencoders #######################################################
# Implement an autoencoder using JAX or Equinox.
# Look at the reconstruction error and the latent space.


# %% Variational Autoencoders ###########################################
# Implement a variational autoencoder using JAX or Equinox.
# Compare the performance of your model to a simple autoencoder.


# %% Bonus ###############################################################
# Take a selfie as your target image.
# Decode a random latent space vector to generate a new face.
# Compute the loss.
# Optimize the latent space vector to minimize the loss.
# Display the optimized latent space vector next to the target image of yourself.
