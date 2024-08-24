# %% utils.py
#     utility functions for aigs
# by: Noah Syrkis

# %% Imports
from jax import lax

# %% Types


# %% Functions
def conv_2d(input, kernel, stride, padding: str):
    return lax.conv(input, kernel, (stride, stride), padding=padding)


def deconv_2d(input, kernel, stride, padding: str):
    return lax.conv_transpose(input, kernel, (stride, stride), padding=padding)
