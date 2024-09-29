import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
import jumanji
# Initialize RNG and environment
rng = jax.random.PRNGKey(0)
env = jumanji.make("Sokoban-v0")


def render_level(state):
    # Render the environment directly
    env.render(state)
    
    # Display a title for the rendered level
    plt.title("Sokoban Level (rendered directly)")
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Define the object type mappings for encoding levels
OBJECT_TYPES = {
    'empty': 0,
    'wall': 1,
    'target': 2,
    'agent': 3,
    'box': 4
}
# Updated function to encode a level into a 3D tensor
def encode_level(grid):

    # Initialize an empty array with shape (height, width, channels)
    encoded = jnp.zeros((10, 10, 5), dtype=jnp.uint8)  # 10x10 grid, 5 channels (empty, wall, target, agent, box)

    # Loop through the level and fill the encoded array
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Check for fixed objects (second index in the grid)
            if grid[i, j, 1] == 1:  # Wall
                encoded = encoded.at[i, j, OBJECT_TYPES['wall']].set(1)
            elif grid[i, j, 1] == 2:  # Goal (target)
                encoded = encoded.at[i, j, OBJECT_TYPES['target']].set(1)

            # Check for movable objects (first index in the grid)
            if grid[i, j, 0] == 3:  # Player/Agent
                encoded = encoded.at[i, j, OBJECT_TYPES['agent']].set(1)
            elif grid[i, j, 0] == 4:  # Box
                encoded = encoded.at[i, j, OBJECT_TYPES['box']].set(1)

    return encoded

# Load and encode multiple levels into 3D tensors
def encode_multiple_levels(num_levels):
    encoded_levels = []
    global rng
    for i in range(num_levels):
        rng, subkey = jax.random.split(rng)

        # Reset the environment to get the initial state
        state, timestep = env.reset(subkey)
        # render_level(state)  # Render the level
        # Encode the level
        encoded_level = encode_level(timestep.observation.grid)  # Use the observation grid
        encoded_levels.append(encoded_level)

    return jnp.stack(encoded_levels)  # Stack into a single 4D tensor


# Now the encoded_levels tensor is available for training
# Encoder
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # Apply convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        # Flatten and project to latent space
        x = x.reshape((x.shape[0], -1))  # Flatten
        latent = nn.Dense(features=self.latent_dim)(x)
        return latent

# Decoder
class Decoder(nn.Module):
    latent_dim: int
    original_shape: tuple

    @nn.compact
    def __call__(self, latent):
        # Project back to a feature map and reshape
        batch_size = latent.shape[0]
        x = nn.Dense(features=128 * (self.original_shape[0] // 4) * (self.original_shape[1] // 4))(latent)
        x = x.reshape((batch_size, self.original_shape[0] // 4, self.original_shape[1] // 4, 128))

        # First transposed convolution to upscale
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        # Second transposed convolution to upscale to 10x10
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        # Ensure the final output has the desired (10, 10, 5) shape
        x = nn.ConvTranspose(features=self.original_shape[2], kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        return x

# Autoencoder (Combining Encoder and Decoder)
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

# Define the loss function (MSE)
def compute_loss(params, model, batch):
    # Forward pass through the model
    reconstructions = model.apply({'params': params}, batch)
    # Mean squared error between original and reconstructed levels
    loss = jnp.mean((reconstructions - batch) ** 2)
    return loss


encoded_levels = encode_multiple_levels(50)
# Optimizer
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)

# Initialize model
original_shape = (10, 10, 5)  # Height x Width x Channels
latent_dim = 32  # Latent space dimensionality
model = Autoencoder(latent_dim=latent_dim, original_shape=original_shape)

# Initialize parameters
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, *original_shape)))['params']

# Initialize optimizer state
opt_state = optimizer.init(params)

# Training step function
@jax.jit
def train_step(params, opt_state, batch):
    # Compute the loss and gradients
    loss, grads = jax.value_and_grad(compute_loss)(params, model, batch)

    # Update the parameters using the gradients
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss

# Training loop
loss_history = []

def train_autoencoder(num_epochs, batch):
    global params, opt_state
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, opt_state, batch)
        loss_history.append(loss)  # Track the loss
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# Prepare the encoded levels as a batch (this is where you use your encoded levels)
# Assuming `encoded_levels` contains the 5 Sokoban levels as 4D tensor (batch_size, height, width, channels)
batch = encoded_levels.reshape((-1, *original_shape))  # Batch size inferred

# Train the model for 100 epochs
train_autoencoder(100, batch)

