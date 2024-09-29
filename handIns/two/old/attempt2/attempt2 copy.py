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
def sample_levels(number_of_levels=5):
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
def train_dense_autoencoder(dataset, latent_dim=32, num_epochs=10, learning_rate=0.001):
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
def generate_levels(autoencoder, params, num_levels=5, latent_dim=32):
    # Generate random latent vectors
    latent_vectors = jax.random.normal(jax.random.PRNGKey(0), (num_levels, latent_dim))
    
    # Decode the latent vectors back to levels using the custom decode method
    generated_levels = autoencoder.apply({'params': params}, latent_vectors, method=autoencoder.decode)
    
    # Clip to [0, 1] to ensure valid image range
    generated_levels = jnp.clip(generated_levels, 0, 1)

    return generated_levels

print('Loading or sampling levels...')
existing_dataset = load_dataset(dataset_path)  # Use load_dataset for normalization
dataset = None

if existing_dataset is None:
    print('Sampling levels...')
    dataset = sample_levels()  # Uncomment to sample levels
else:
    dataset = existing_dataset  # Keep existing dataset
    print('Loaded existing dataset.')

# Save the normalized dataset
save_data(dataset_path, dataset)

print(f'Dataset shape: {dataset.shape}')

# Train the dense autoencoder if dataset exists
# if dataset is not None and dataset.shape[0] > 0:
#     print('Training the dense autoencoder...')
#     autoencoder, params = train_dense_autoencoder(dataset)  # Uncomment to train
#     # Save parameters after training
#     save_data(params_path, params)
# else:
#     print("No dataset available for training.")

# Load the trained model parameters
params = load_data(params_path)

# Make sure to initialize the autoencoder with correct input dimension
input_dim = np.prod(dataset.shape[1:])  # Ensure this is correct
autoencoder = DenseAutoencoder(latent_dim=32, input_dim=input_dim)  # Correctly initialize

# Generate and display new levels
print('Generating new levels...')
generated_levels = generate_levels(autoencoder, params, num_levels=5)

# Convert generated levels back to original shape and display
for i in range(generated_levels.shape[0]):
    generated_level_image = generated_levels[i].reshape(dataset.shape[1:])  # Reshape to original level dimensions
    display_original_and_decoded(np.zeros_like(generated_level_image), generated_level_image)  # Original is blank
