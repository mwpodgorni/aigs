# %% vae.py
#   (variational) autoencoder code for aigs
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import random, tree, grad, nn
import chex
