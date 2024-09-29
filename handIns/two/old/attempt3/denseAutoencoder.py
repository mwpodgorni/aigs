import jax
import jumanji
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np

# Initialize RNG and environment
rng = jax.random.PRNGKey(0)
env = jumanji.make("Sokoban-v0")

# Define the object type mappings
OBJECT_TYPES = {
    'empty': 0,
    'wall': 1,
    'goal': 2,
    'player': 3,
    'box': 4
}

# Function to encode a level into a 3D tensor
def encode_level(grid):
    encoded = jnp.zeros((10, 10, 5), dtype=jnp.uint8)  # 10x10 grid, 5 channels
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Fixed objects (walls and targets)
            if grid[i, j, 0] == 1:  # Wall
                encoded = encoded.at[i, j, OBJECT_TYPES['wall']].set(1)
            elif grid[i, j, 0] == 2:  # Target
                encoded = encoded.at[i, j, OBJECT_TYPES['goal']].set(1)
            
            # Movable objects (agent and box)
            if grid[i, j, 1] == 3:  # Player (agent)
                encoded = encoded.at[i, j, OBJECT_TYPES['player']].set(1)
            elif grid[i, j, 1] == 4:  # Box
                encoded = encoded.at[i, j, OBJECT_TYPES['box']].set(1)

    return encoded

# Load and encode 5 levels into 3D tensors
# Function to encode multiple levels
def encode_multiple_levels(num_levels):
    encoded_levels = []
    for i in range(num_levels):
        global rng
        rng, subkey = jax.random.split(rng)
        state, timestep = env.reset(subkey)
        
        # Print the raw grid from the environment
        # print(f"Raw Grid {i}:\n", timestep.observation.grid)
        
        encoded_level = encode_level(timestep.observation.grid)
        # print(f"Encoded Level {i}:\n", encoded_level)  # Print encoded level for comparison
        encoded_levels.append(encoded_level)
    return jnp.stack(encoded_levels)

# Define the dense autoencoder model
class DenseAutoencoder(nn.Module):
    latent_dim: int = 8  # Latent dimension

    def setup(self):
        self.encoder = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,  # Use leaky relu
            nn.Dense(self.latent_dim),
        ])
        self.decoder = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,  # Use leaky relu
            nn.Dense(10 * 10 * 5),  # Output size
            nn.sigmoid,  # Apply sigmoid to output
        ])

    def encode(self, x):
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten the input
        return self.encoder(x)

    def decode(self, z):
        z = self.decoder(z)
        return jnp.reshape(z, (z.shape[0], 10, 10, 5))  # Reshape back to original shape

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

# Train the autoencoder
def train_autoencoder(encoded_levels):
    model = DenseAutoencoder()
    params = model.init(rng, jnp.ones((1, 10, 10, 5), dtype=jnp.float32))  # Dummy input for init
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    for epoch in range(100):  # Adjust number of epochs
        loss_fn = lambda p: jnp.mean((model.apply(p, encoded_levels) - encoded_levels) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return params, model

# Function to decode and display results
# Function to decode and display results
def decode_and_display(encoded_levels, model, params):
    decoded_levels = model.apply(params, encoded_levels)

    # Check the shape and value range of the decoded output
    print("Decoded Levels Shape:", decoded_levels.shape)
    print("Decoded Levels Min/Max:", decoded_levels.min(), decoded_levels.max())

    # Iterate through each decoded level and compare with the original
    for i in range(decoded_levels.shape[0]):
        # Create a figure for the original and decoded levels
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f'Original vs Decoded Level {i + 1}', fontsize=16)

        # Plot each channel of the original and decoded levels side by side
        for channel in range(decoded_levels.shape[-1]):
            # Original level
            original_channel = encoded_levels[i, :, :, channel]
            axes[0, channel].imshow(original_channel, vmin=0, vmax=1)
            axes[0, channel].set_title(f'Original Channel {channel}')
            axes[0, channel].axis('off')

            # Decoded level
            decoded_channel = decoded_levels[i, :, :, channel]
            # Binarize the decoded channel (since you have binary encoding for objects)
            decoded_channel_binarized = (decoded_channel > 0.5).astype(np.float32)
            axes[1, channel].imshow(decoded_channel_binarized, vmin=0, vmax=1)
            axes[1, channel].set_title(f'Decoded Channel {channel}')
            axes[1, channel].axis('off')

        plt.show()

# Main execution
encoded_levels = encode_multiple_levels(5)
print("Encoded Levels Shape:", encoded_levels.shape)

# Train the autoencoder
trained_params, autoencoder_model = train_autoencoder(encoded_levels)

# Decode and display results (original vs decoded)
decode_and_display(encoded_levels, autoencoder_model, trained_params)
