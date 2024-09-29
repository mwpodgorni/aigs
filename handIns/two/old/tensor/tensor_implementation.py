import gym
import gym_sokoban
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

# Create the Sokoban environment
env = gym.make('Sokoban-v2')

def sample_levels(env, number_of_levels=5):
    print('Sampling levels...')
    levels = []
    for _ in tqdm(range(number_of_levels)):
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            levels.append(observation)
    return np.array(levels)

# Define paths for saving/loading
dataset_path = 'sokoban_levels.pkl'
model_path = 'autoencoder_weights.pkl'

# Load existing levels or sample new ones
if os.path.exists(dataset_path):
    with open(dataset_path, 'rb') as f:
        existing_levels = pickle.load(f)
    print(f'Loaded {len(existing_levels)} existing levels.')
else:
    existing_levels = np.array([])

# Sample new levels
new_levels = sample_levels(env)
# print(f'Sampled {len(new_levels)} new levels.')
# Combine existing levels with new levels if existing ones are available
if existing_levels.size > 0:
    # levels = np.concatenate((existing_levels, new_levels), axis=0)
    levels=existing_levels
else:
    levels = new_levels

# Save combined levels
with open(dataset_path, 'wb') as f:
    pickle.dump(levels, f)
print('-')
print('-')
print('-')
print(f'Total {len(levels)} levels saved with shape: {levels.shape}')
env.close()

# Preprocess the levels
def preprocess_levels(levels):
    levels = levels.astype(np.float32) / 255.0
    return levels

processed_levels = preprocess_levels(levels)

# Build the autoencoder
def build_autoencoder(input_shape):
    encoder = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2, input_shape=input_shape),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        layers.Flatten(),
        layers.Dense(latent_dim)
    ])

    decoder = keras.Sequential([
        layers.Dense(np.prod((height // 4, width // 4, 8)), activation='relu', input_shape=(latent_dim,)),
        layers.Reshape((height // 4, width // 4, 8)),
        layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(channels, (3, 3), activation='sigmoid', padding='same', strides=2)
    ])

    return encoder, decoder

# Define input shape and latent dimension
height, width, channels = processed_levels.shape[1:4]
latent_dim = 64
encoder, decoder = build_autoencoder((height, width, channels))

# Combine encoder and decoder into an autoencoder
autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Load model weights if available
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
        # autoencoder.set_weights(weights)
else:
    print("No saved model found. Starting training from scratch.")

# Train the autoencoder
autoencoder.fit(processed_levels, processed_levels, epochs=100, batch_size=32)

# Save model weights after training
with open(model_path, 'wb') as f:
    pickle.dump(autoencoder.get_weights(), f)

# Generate a new level
def generate_level(autoencoder, latent_dim):
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_level = decoder.predict(random_latent_vector)
    return generated_level

generated_level = generate_level(autoencoder, latent_dim)

# Display levels
def display_levels(original, generated):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original)
    ax[0].set_title('Original Level')
    ax[0].axis('off')

    ax[1].imshow(generated)
    ax[1].set_title('Generated Level')
    ax[1].axis('off')

    plt.show()

# Display an original level and the generated level
display_levels(processed_levels[0], np.clip(generated_level.squeeze(), 0, 1))
