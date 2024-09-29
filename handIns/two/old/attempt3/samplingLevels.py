import jax
import jumanji
import jax.numpy as jnp

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
    # Initialize an empty array with shape (height, width, channels)
    encoded = jnp.zeros((10, 10, 5), dtype=jnp.uint8)  # 10x10 grid, 5 channels

    # Loop through the level and fill the encoded array
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Check the first channel (fixed objects)
            if grid[i, j, 0] == 1:  # Wall
                encoded = encoded.at[i, j, OBJECT_TYPES['wall']].set(1)
            elif grid[i, j, 0] == 2:  # Target
                encoded = encoded.at[i, j, OBJECT_TYPES['goal']].set(1)

            # Check the second channel (movable objects)
            if grid[i, j, 1] == 3:  # Player
                encoded = encoded.at[i, j, OBJECT_TYPES['player']].set(1)
            elif grid[i, j, 1] == 4:  # Box
                encoded = encoded.at[i, j, OBJECT_TYPES['box']].set(1)
    return encoded

# Load and encode 5 levels into 3D tensors
def encode_multiple_levels(num_levels):
    encoded_levels = []
    for i in range(num_levels):
        global rng
        rng, subkey = jax.random.split(rng)

        # Reset the environment to get the initial state
        state, timestep = env.reset(subkey)

        # Encode the level
        encoded_level = encode_level(timestep.observation.grid)  # Use the observation grid
        encoded_levels.append(encoded_level)

    return jnp.stack(encoded_levels)  # Stack into a single 4D tensor

# Encode 5 levels
encoded_levels = encode_multiple_levels(5)

# Print the shape of the encoded levels
print("Encoded Levels Shape:", encoded_levels.shape) 