# %% Import necessary libraries
import gym
import gym_sokoban
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import jax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
import os  # Import for file existence check
import pickle  # Import for saving/loading data

# Instantiate a Gym Sokoban environment
env = gym.make('Sokoban-v2')
dataset_path = 'sokoban_levels.npy'
params_path = 'autoencoder_params.pkl'
opt_state_path = 'optimizer_state.pkl'

print('start')

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
        output_width = self.width // 8    # After three convolutions with stride 2

        # Set the dense layer input size correctly
        self.dense1 = nn.Dense(self.latent_dim)
        self.dense2 = nn.Dense(64 * output_height * output_width)  # Flattened size

        self.conv_trans1 = nn.ConvTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.conv_trans2 = nn.ConvTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.conv_trans3 = nn.ConvTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='SAME')

    def encode(self, x):
        x = self.conv1(x)
        x = nn.relu(x)

        x = self.conv2(x)
        x = nn.relu(x)

        x = self.conv3(x)
        x = nn.relu(x)

        # Flatten the output
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten
        return self.dense1(x)

    def decode(self, z):
        x = self.dense2(z)
        x = x.reshape(-1, self.height // 8, self.width // 8, 64)  # Reshape based on last conv output
        x = self.conv_trans1(x)
        x = nn.relu(x)
        x = self.conv_trans2(x)
        x = nn.relu(x)
        return nn.sigmoid(self.conv_trans3(x))  # Output RGB

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

# %% Function to sample levels
def sample_levels(number_of_levels=10):
    levels = []
    for _ in range(number_of_levels):
        observation = env.reset()
        if observation is not None:
            levels.append(observation)
    return np.array(levels)

# %% Function to save and load data
def save_data(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# Separate function for loading the dataset with normalization
def load_dataset(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data / 255.0  # Normalize to [0, 1]
    return None

# Function for loading model parameters and optimizer state
def load_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# %% Training loop function with saving/loading
def train_autoencoder(dataset, latent_dim=32, num_epochs=50, learning_rate=0.001):
    height, width, channels = dataset.shape[1:4]
    autoencoder = Autoencoder(latent_dim=latent_dim, height=height, width=width)
    rng = jax.random.PRNGKey(0)

    # Check if saved parameters exist, otherwise initialize
    params = load_data(params_path) or autoencoder.init(rng, jnp.ones((1, height, width, channels)))  # Dummy input for shape
    opt_state = load_data(opt_state_path)
    optimizer = optax.adam(learning_rate)
    if opt_state is None:
        opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x):
        x_reconstructed = autoencoder.apply(params, x)
        return jnp.mean((x - x_reconstructed) ** 2)

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for x in dataset:
            x = jnp.array(x)
            x = x[jnp.newaxis, ...]
            loss, grads = jax.value_and_grad(loss_fn)(params, x)

            # Print gradient norms
            grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
            # print(f"Epoch {epoch + 1} Gradient Norm: {grad_norm:.4f}")

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}')

        if (epoch + 1) % 50 == 0:
            save_data(params_path, params)
            save_data(opt_state_path, opt_state)
            print('Model parameters and optimizer state saved.')

    return autoencoder, params

# %% Function to visualize original and decoded levels side by side
def display_original_and_decoded(original, decoded):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original)
    ax[0].set_title('Original Level')
    ax[0].axis('off')

    ax[1].imshow(decoded)
    ax[1].set_title('Decoded Level')
    ax[1].axis('off')

    plt.show()

# %% Load dataset if it exists, otherwise sample and save
print('Loading or sampling levels...')
existing_dataset = load_dataset(dataset_path)  # Use load_dataset for normalization
new_levels = sample_levels()

# Normalize the new levels before saving
new_levels = new_levels / 242.0  # Normalize to [0, 1]

dataset = None
if existing_dataset is None:
    print('Sampling levels...')
    dataset = new_levels
else:
    dataset = np.concatenate((existing_dataset, new_levels), axis=0)
    print('Loaded existing dataset.')
# print("Original Min pixel value:", jnp.min(dataset[0]))
# print("Original Max pixel value:", jnp.max(dataset[0]))
# Save the normalized dataset
save_data(dataset_path, dataset)

print(f'Dataset shape: {dataset.shape}')

# Train the autoencoder
print('Training the autoencoder...')
autoencoder, params = train_autoencoder(dataset)

# Generate levels and display them
original = jnp.array(dataset[-1])[jnp.newaxis, ...]  # Add batch dimension for the model input
decoded = autoencoder.apply(params, original)  # Get the decoded output
decoded = jnp.clip(decoded, 0, 1)  # Clip the values to be in the range [0, 1]

# Display the last original and decoded level
display_original_and_decoded(original.squeeze(), decoded.squeeze())
