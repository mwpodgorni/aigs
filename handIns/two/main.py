# %% Import necessary libraries
import gym
import gym_sokoban
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import optax
import jax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
# %% Constants for object types
EMPTY_SPACE_VALUE = [0, 0, 0]      # Typically black (RGB)
WALL_VALUE = [100, 100, 100]        # Typically a shade of gray
TARGET_VALUE = [255, 255, 255]      # Typically white
AGENT_VALUE = [150, 150, 150]       # Light gray for the agent
BOX_VALUE_1 = [215, 103, 0]   
BOX_VALUE_2 = [215, 114, 0]
BOX_VALUE_3 = [215, 196, 0]
# Instantiate a Gym Sokoban environment
env = gym.make('Sokoban-v2')


# %% Define the Autoencoder class
class Autoencoder(nn.Module):
    latent_dim: int
    height: int
    width: int

    def setup(self):
        # Define convolutional layers
        self.conv1 = nn.Conv(16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.conv2 = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.conv3 = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')

        # Calculate output size after the convolutional layers
        output_height = self.height // 8  # After three convolutions with stride 2
        output_width = self.width // 8      # After three convolutions with stride 2

        # Set the dense layer input size correctly
        self.dense1 = nn.Dense(self.latent_dim)
        self.dense2 = nn.Dense(64 * output_height * output_width)  # Flattened size

        self.conv_trans1 = nn.ConvTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.conv_trans2 = nn.ConvTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.conv_trans3 = nn.ConvTranspose(5, kernel_size=(3, 3), strides=(2, 2), padding='SAME')

    def encode(self, x):
        print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = nn.relu(x)

        x = self.conv2(x)
        print(f"After conv2: {x.shape}")
        x = nn.relu(x)

        x = self.conv3(x)
        print(f"After conv3: {x.shape}")
        x = nn.relu(x)

        # Flatten the output
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten
        print(f"Flattened shape: {x.shape}")
        return self.dense1(x)

    def decode(self, z):
        x = self.dense2(z)
        x = x.reshape(-1, self.height // 8, self.width // 8, 64)  # Reshape based on last conv output
        x = self.conv_trans1(x)
        x = nn.relu(x)
        x = self.conv_trans2(x)
        x = nn.relu(x)
        return self.conv_trans3(x)

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

# %% Training loop function
def train_autoencoder(dataset, latent_dim=32, num_epochs=10, learning_rate=0.001):
    height, width, channels = dataset.shape[1:4]
    autoencoder = Autoencoder(latent_dim=latent_dim, height=height, width=width)
    rng = jax.random.PRNGKey(0)

    # Initialize the model parameters
    params = autoencoder.init(rng, jnp.ones((1, height, width, channels)))  # Use a dummy input for shape

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Define the loss function
    def loss_fn(params, x):
        x_reconstructed = autoencoder.apply(params, x)
        return jnp.mean((x - x_reconstructed) ** 2)

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for x in dataset:
            x = jnp.array(x)  # Ensure input is a JAX array
            x = x[jnp.newaxis, ...]  # Add a new axis for batch size (1, height, width, channels)
            loss, grads = jax.value_and_grad(loss_fn)(params, x)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}')

# %% Function to sample levels
def sample_levels(num_levels):
    levels = []
    for _ in tqdm(range(num_levels)):
        observation = env.reset()
        # Ensure observation is of the expected shape
        if observation is not None:
            level_tensor = convert_to_tensor(observation)
            levels.append(level_tensor)
    return np.array(levels)
# %% 
def display_observation(observation):
    plt.imshow(observation)
    plt.title('Sokoban Level Observation')
    plt.axis('off')  
    plt.show()

# %% 
def convert_to_tensor(level_layout):
    height, width, _ = level_layout.shape  # Assuming level_layout is a 3D image
    num_channels = 5  # Now 5 channels to represent each object type
    
    tensor = np.zeros((height, width, num_channels))
    # print("Unique values in level_layout[:,:,0]:", np.unique(level_layout[:, :, 0]))  # Check for unique values in the layout

    # Populate each channel based on object types in the level_layout
    tensor[:, :, 0] = np.all(level_layout == EMPTY_SPACE_VALUE, axis=-1) 
    tensor[:, :, 1] = np.all(level_layout == WALL_VALUE, axis=-1)        
    tensor[:, :, 2] = np.all(level_layout == TARGET_VALUE, axis=-1)      
    tensor[:, :, 3] = np.all(level_layout == AGENT_VALUE, axis=-1)       
    tensor[:, :, 4] = np.any([np.all(level_layout == BOX_VALUE_1, axis=-1), 
                            np.all(level_layout == BOX_VALUE_2, axis=-1),
                            np.all(level_layout == BOX_VALUE_2, axis=-1)], axis=0)      
    
    return tensor

# Function to visualize the converted level
# %% 
def visualize_level(level_tensor):
    # Create a 2D representation for visualization
    height, width, _ = level_tensor.shape
    image = np.zeros((height, width), dtype=np.uint8)

    # Map each channel to a unique grayscale value
    image[level_tensor[:, :, 0] == 1] = 255  # Empty Space (white)
    image[level_tensor[:, :, 1] == 1] = 0    # Walls (black)
    image[level_tensor[:, :, 2] == 1] = 200  # Targets (gray)
    image[level_tensor[:, :, 3] == 1] = 150  # Agent (light gray)
    image[level_tensor[:, :, 4] == 1] = 100  # Boxes (darker gray)

    # Display the image
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Turn off axis labels
    plt.show()

# %% Sample levels and display them
dataset = sample_levels(2)
print('Dataset collected.')
print(f'Dataset shape: {dataset.shape}') 
# visualize_level(dataset[0])

# Train the autoencoder
train_autoencoder(dataset)



# %% Optionally save levels to a pickle file
# with open('sokoban_levels.pkl', 'wb') as f:
#     pickle.dump(dataset, f)
# print("Levels saved to 'sokoban_levels.pkl'.")
# %%


# def analyze_observation(observation):
#     unique_values = np.unique(observation.reshape(-1, observation.shape[2]), axis=0)
#     print("Unique RGB values in observation:", unique_values)

# # Add this call after resetting the environment to inspect the RGB values.
# sample_observation = env.reset()
# analyze_observation(sample_observation)