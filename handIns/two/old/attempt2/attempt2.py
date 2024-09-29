# %% Import necessary libraries
import gym
import gym_sokoban
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import jax
from flax import linen as nn
from tqdm import tqdm
import os  # Import for file existence check
import pickle  # Import for saving/loading data

# Instantiate a Gym Sokoban environment
env = gym.make('Sokoban-v2')
dataset_path = 'sokoban_levels.npy'
params_path = 'autoencoder_params.pkl'
opt_state_path = 'optimizer_state.pkl'

print('start')

# Define the Dense Autoencoder class
class DenseAutoencoder(nn.Module):
    latent_dim: int
    input_dim: int

    def setup(self):
        # Define the dense layers for encoding
        self.encoder = nn.Dense(self.latent_dim)  # Latent space
        self.decoder = nn.Dense(self.input_dim)   # Reconstructed output

    def __call__(self, x):
        z = self.encoder(x)  # Encode
        return nn.sigmoid(self.decoder(z))  # Decode

# %% Function to sample levels
def sample_levels(number_of_levels=20):
    levels = []
    for _ in tqdm(range(number_of_levels)):
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
def train_dense_autoencoder(dataset, latent_dim=32, num_epochs=50, learning_rate=0.001):
    # Flatten the input dataset to match the dense layer requirements
    input_dim = np.prod(dataset.shape[1:])  # Height * Width * Channels
    dataset_flat = dataset.reshape(dataset.shape[0], -1)

    autoencoder = DenseAutoencoder(latent_dim=latent_dim, input_dim=input_dim)
    rng = jax.random.PRNGKey(0)

    # Check if saved parameters exist, otherwise initialize
    params = load_data(params_path) or autoencoder.init(rng, jnp.ones((1, input_dim)))  # Dummy input for shape
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
        for x in dataset_flat:  # Use flattened dataset
            x = jnp.array(x)
            x = x[jnp.newaxis, ...]
            loss, grads = jax.value_and_grad(loss_fn)(params, x)

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataset_flat):.4f}')

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
new_levels = new_levels / 255.0  # Normalize to [0, 1]

dataset = None
if existing_dataset is None:
    print('Sampling levels...')
    dataset = new_levels
else:
    dataset = np.concatenate((existing_dataset, new_levels), axis=0)
    print('Loaded existing dataset.')

# Save the normalized dataset
save_data(dataset_path, dataset)

print(f'Dataset shape: {dataset.shape}')

# Train the dense autoencoder
print('Training the dense autoencoder...')
autoencoder, params = train_dense_autoencoder(dataset)

# Display original and decoded levels after training
print('Displaying original and decoded levels...')
sample_index = np.random.randint(0, dataset.shape[0])  # Randomly select an index
original_level = dataset[sample_index]
original_level_flat = jnp.array(original_level).flatten()[jnp.newaxis, ...]  # Flatten for the model

# Get the decoded output from the autoencoder
decoded_level = autoencoder.apply(params, original_level_flat)

# Convert decoded level back to image format (assuming sigmoid outputs values in [0, 1])
decoded_level_image = jnp.clip(decoded_level, 0, 1).reshape(original_level.shape)

# Display the original and decoded level side by side
display_original_and_decoded(original_level, decoded_level_image)
