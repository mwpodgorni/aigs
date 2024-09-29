# %% Import necessary libraries
import gym
import gym_sokoban
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import optax
import jax
import os
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
        # print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = nn.relu(x)

        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        x = nn.relu(x)

        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        x = nn.relu(x)

        # Flatten the output
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten
        # print(f"Flattened shape: {x.shape}")
        return self.dense1(x)

    def decode(self, z):
        x = self.dense2(z)
        x = x.reshape(-1, self.height // 8, self.width // 8, 64)  # Reshape based on last conv output
        x = self.conv_trans1(x)
        x = nn.relu(x)
        x = self.conv_trans2(x)
        x = nn.relu(x)
        x = self.conv_trans3(x)
        
        # Constrain the output between 0 and 1 (to represent probabilities for each object type)
        return nn.sigmoid(x)

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

# %% Save and load functions
def save_dataset(dataset, filename='dataset.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Dataset saved to {filename}.')

def load_dataset(filename='dataset.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        print(f'Dataset loaded from {filename}.')
        return dataset
    else:
        print(f'No dataset found at {filename}.')
        return None

def save_model_params(params, filename='model_params.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f'Model parameters saved to {filename}.')

def load_model_params(filename='model_params.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        print(f'Model parameters loaded from {filename}.')
        return params
    else:
        print(f'No model parameters found at {filename}.')
        return None

# %% Training loop function
def train_autoencoder(dataset, latent_dim=32, num_epochs=10, learning_rate=0.001):
    height, width, channels = dataset.shape[1:4]
    autoencoder = Autoencoder(latent_dim=latent_dim, height=height, width=width)
    rng = jax.random.PRNGKey(0)

    params = autoencoder.init(rng, jnp.ones((1, height, width, channels)))  # Use a dummy input for shape

    # Load model parameters if available
    loaded_params = load_model_params()
    if loaded_params is not None:
        params = loaded_params

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
        for i, x in enumerate(dataset):
            x = jnp.array(x)  # Ensure input is a JAX array
            x = x[jnp.newaxis, ...]  # Add a new axis for batch size (1, height, width, channels)
            loss, grads = jax.value_and_grad(loss_fn)(params, x)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss
            
            # Save progress every 50 iterations
            if (i + 1) % 50 == 0:
                save_model_params(params)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}')

    # Save final parameters after training
    save_model_params(params)
    return autoencoder, params

# %% Function to sample levels
def sample_levels(num_levels):
    print('sample_levels')
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
    
    # Initialize a tensor to hold colored levels
    tensor = np.zeros((height, width, 3), dtype=np.uint8)  # RGB format

    # Populate the tensor with original colors
    for y in range(height):
        for x in range(width):
            if np.all(level_layout[y, x] == EMPTY_SPACE_VALUE):
                tensor[y, x] = [0, 0, 0]  # Empty Space (black)
            elif np.all(level_layout[y, x] == WALL_VALUE):
                tensor[y, x] = [100, 100, 100]  # Walls (gray)
            elif np.all(level_layout[y, x] == TARGET_VALUE):
                tensor[y, x] = [255, 255, 255]  # Targets (white)
            elif np.all(level_layout[y, x] == AGENT_VALUE):
                tensor[y, x] = [150, 150, 150]  # Agent (light gray)
            elif np.any([np.all(level_layout[y, x] == BOX_VALUE_1),
                          np.all(level_layout[y, x] == BOX_VALUE_2),
                          np.all(level_layout[y, x] == BOX_VALUE_3)]):
                tensor[y, x] = [215, 196, 0]  # Boxes (orange-like color)

    return tensor

# %% Sample levels and display them
existing_dataset  = load_dataset()  # Try to load existing dataset
# new_levels = sample_levels(30) 
if existing_dataset  is None:  # If no dataset is loaded, sample new levels
    dataset = existing_dataset 
# else:
#     print(f'Dataset shape: {dataset.shape}') 
else:
    # Combine existing levels with new levels
    dataset = existing_dataset 
    # dataset = np.concatenate((existing_dataset, new_levels), axis=0)
    
save_dataset(dataset) 
# Train the autoencoder
autoencoder, params = train_autoencoder(dataset)

# Function to sample random latent vectors and generate levels
def generate_new_levels(autoencoder, params, num_samples=10, latent_dim=32):
    # Sample random points from a standard normal distribution in the latent space
    rng = jax.random.PRNGKey(42)  # Use a fixed seed for reproducibility
    z_samples = jax.random.normal(rng, (num_samples, latent_dim))

    # Decode the latent vectors to generate new levels
    # Decode the latent vectors to generate new levels
    decoded_levels = autoencoder.apply(params, z_samples, method=autoencoder.decode)

    # Post-process: convert sigmoid outputs back to binary (channel-based) Sokoban level maps
    generated_levels = (decoded_levels > 0.5).astype(jnp.float32)  # Binary output

    return generated_levels

# Function to visualize generated levels (same as before)
def visualize_level(level_tensor):
    height, width, _ = level_tensor.shape
    # Create an empty image with RGB channels
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign original colors based on channels
    image[level_tensor[:, :, 0] == 1] = EMPTY_SPACE_VALUE   # Empty Space
    image[level_tensor[:, :, 1] == 1] = WALL_VALUE         # Walls
    image[level_tensor[:, :, 2] == 1] = TARGET_VALUE       # Targets
    image[level_tensor[:, :, 3] == 1] = AGENT_VALUE        # Agent
    image[level_tensor[:, :, 4] == 1] = BOX_VALUE_1        # Boxes (you can choose one box color or mix them)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Function to display multiple generated levels
def display_generated_levels(generated_levels):
    for i, level in enumerate(generated_levels):
        print(f"Displaying generated level {i + 1}:")
        visualize_level(np.array(level))

# After training the autoencoder, generate and display new levels
generated_levels = generate_new_levels(autoencoder, params, num_samples=5)
display_generated_levels(generated_levels)