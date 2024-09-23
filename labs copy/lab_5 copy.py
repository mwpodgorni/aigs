# %% Imports
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Step 1: Get sample images from the environment
def get_sample_images():
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    states = []
    state, _ = env.reset()
    for i in tqdm(range(1500)):
        action = env.action_space.sample()
        state, *_ = env.step(action)
        if i >= 500:
            states.append(state)
    return jnp.array(states).astype(jnp.float32) / 255.0  # Normalize

data = get_sample_images()

plt.imshow(data[500])
plt.show()

# %% Step 2: Encoder and Decoder
class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        return x

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, 12, 12, 128))
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2))(x)
        return x

# %% Step 3: Autoencoder
class Autoencoder(nn.Module):
    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def __call__(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def create_train_state(rng, learning_rate):
    model = Autoencoder()
    variables = model.init(rng, jnp.ones((1, 96, 96, 3)))
    params = variables['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        reconstructed = state.apply_fn({'params': params}, batch)
        loss = jnp.mean((reconstructed - batch) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# %% Step 4: Initialize training state
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate=1e-3)

# %% Step 5: Training loop
for epoch in tqdm(range(10)):  # Adjust the number of epochs as needed
    state, loss = train_step(state, data)
    print(f'Epoch {epoch}, Loss: {loss}')

# %% Step 6: Reconstruct images
reconstructed_images = state.apply_fn({'params': state.params}, data)

# %% Step 7: Display original and reconstructed images
# Clipping the image data before displaying
clipped_data = jnp.clip(data, 0, 1)
clipped_reconstructed_images = jnp.clip(reconstructed_images, 0, 1)

# Adjust the number of columns based on the number of images
fig, axes = plt.subplots(2, min(10, len(clipped_data)), figsize=(10, 10))

# Display original and reconstructed images
for i in range(min(10, len(clipped_data))):
    axes[0, i].imshow(clipped_data[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(clipped_reconstructed_images[i])
    axes[1, i].axis('off')

plt.show()
