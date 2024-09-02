# %% lab_4.py
#    deep learning with JAX
# by: Noah Syrkis

# %% Imports ############################################################
from jax import grad, lax, random
import jax.numpy as jnp
from jaxtyping import Array
import chex
from typing import Callable, Dict, List
import sklearn.datasets as skd
import tensorflow_datasets as tfds
import seaborn as sns
import matplotlib.pyplot as plt

# %%
mnist = tfds.load("mnist", split="train")
x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1, 1, 28, 28)
y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

# %% Convolutional neural networks ######################################
# Make a CNN classifying faces using JAX or Equinox.
# Compare the performance of your model to a simple neural network.
def conv(input, kernel):
    return lax.conv(input, kernel, (1, 1), padding='SAME')

rng = random.PRNGKey(0)
# 1 channel to 4 channels
kernel = random.normal(rng, (4, 1, 3, 3))  # <- 3x3 kernel, 1 input channel, 4 output channels
output = conv(x_data, kernel)  # <- x.data.shape = (n, 1, 28, 28)

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
