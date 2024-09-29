import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
import jumanji
# Initialize RNG and environment
rng = jax.random.PRNGKey(0)
env = jumanji.make("Sokoban-v0")

# Define the object type mappings for encoding levels
OBJECT_TYPES = {
    'empty': 0,
    'wall': 1,
    'target': 2,
    'agent': 3,
    'box': 4
}

# Function to encode a level into a 3D tensor
def encode_level(grid):
    encoded = jnp.zeros((10, 10, 5), dtype=jnp.uint8)  # 10x10 grid, 5 channels

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j, 1] == 1:  # Wall
                encoded = encoded.at[i, j, OBJECT_TYPES['wall']].set(1)
            elif grid[i, j, 1] == 2:  # Target
                encoded = encoded.at[i, j, OBJECT_TYPES['target']].set(1)

            if grid[i, j, 0] == 3:  # Player
                encoded = encoded.at[i, j, OBJECT_TYPES['agent']].set(1)
            elif grid[i, j, 0] == 4:  # Box
                encoded = encoded.at[i, j, OBJECT_TYPES['box']].set(1)

    return encoded

# Encode multiple levels into 3D tensors
def encode_multiple_levels(num_levels):
    encoded_levels = []
    global rng
    for i in range(num_levels):
        rng, subkey = jax.random.split(rng)
        state, timestep = env.reset(subkey)
        encoded_level = encode_level(timestep.observation.grid)
        encoded_levels.append(encoded_level)

    return jnp.stack(encoded_levels)

# Simplified Encoder
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        latent = nn.Dense(features=self.latent_dim)(x)
        return latent

# Simplified Decoder
class Decoder(nn.Module):
    latent_dim: int
    original_shape: tuple

    @nn.compact
    def __call__(self, latent):
        batch_size = latent.shape[0]
        x = nn.Dense(features=128 * (self.original_shape[0] // 4) * (self.original_shape[1] // 4))(latent)
        x = x.reshape((batch_size, self.original_shape[0] // 4, self.original_shape[1] // 4, 128))

        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=self.original_shape[2], kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        return x

# Autoencoder combining Encoder and Decoder
class Autoencoder(nn.Module):
    latent_dim: int
    original_shape: tuple

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.original_shape)

    def __call__(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# MSE Loss
# Define the cross-entropy loss function
def compute_loss(params, model, batch):
    # Forward pass through the model
    reconstructions = model.apply({'params': params}, batch)
    # Use softmax cross-entropy for classification
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=reconstructions, 
        labels=jnp.argmax(batch, axis=-1)
    )
    return jnp.mean(loss)

# Initialize model, optimizer, and parameters
encoded_levels = encode_multiple_levels(50)
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
original_shape = (10, 10, 5)
latent_dim = 64  # Keep latent dimension small for quick training
model = Autoencoder(latent_dim=latent_dim, original_shape=original_shape)
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, *original_shape)))['params']
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(compute_loss)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# Train the model
def train_autoencoder(num_epochs, batch):
    global params, opt_state
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, opt_state, batch)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

batch = encoded_levels.reshape((-1, *original_shape))
train_autoencoder(1000, batch)  # Reduce epochs for faster training

# Visualization of Decoded Levels
def visualize_decoded_level(model, params, original_level):
    original_level = original_level.reshape(1, *original_shape)
    reconstructed = model.apply({'params': params}, original_level).squeeze()

    original_level_classes = jnp.argmax(original_level.squeeze(), axis=-1)
    reconstructed_level_classes = jnp.argmax(reconstructed, axis=-1)

    color_map = {
        0: [255, 255, 255],  # empty -> white
        1: [139, 69, 19],    # wall -> brown
        2: [255, 215, 0],    # target -> yellow
        3: [0, 255, 0],      # agent -> green
        4: [255, 0, 0]       # box -> red
    }

    def map_to_rgb(level_classes):
        rgb_image = jnp.zeros((level_classes.shape[0], level_classes.shape[1], 3), dtype=jnp.uint8)
        for object_type, color in color_map.items():
            mask = (level_classes == object_type)
            for channel in range(3):
                rgb_image = rgb_image.at[mask, channel].set(color[channel])
        return rgb_image

    original_rgb = map_to_rgb(original_level_classes)
    reconstructed_rgb = map_to_rgb(reconstructed_level_classes)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Level")
    axs[0].axis('off')
    axs[1].imshow(reconstructed_rgb)
    axs[1].set_title("Reconstructed Level")
    axs[1].axis('off')
    plt.show()

visualize_decoded_level(model, params, encoded_levels[-1])
