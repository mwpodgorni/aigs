import gym
import gym_sokoban
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
# Create the Sokoban environment
env = gym.make('Sokoban-v2')

def sample_levels(env, number_of_levels=10):
    levels = []
    for _ in tqdm(range(number_of_levels)):
        observation = env.reset()  # Reset the environment to get the initial state
        done = False
        while not done:
            action = env.action_space.sample()  # Randomly sample an action
            observation, reward, done, info = env.step(action)  # Take the action
            levels.append(observation)  # Save the observation (level)
    return np.array(levels)

# Sample 1000 levels
levels = sample_levels(env, number_of_levels=50)
print(f'Sampled {len(levels)} levels with shape: {levels.shape}')

# Close the environment
env.close()

def preprocess_levels(levels):
    # Normalize the levels to be between 0 and 1
    levels = levels.astype(np.float32) / 255.0  # Assuming original levels are in 0-255 range
    return levels

# Preprocess the levels
processed_levels = preprocess_levels(levels)

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
        layers.Conv2DTranspose(channels, (3, 3), activation='sigmoid', padding='same', strides=2)  # Set to channels
    ])

    return encoder, decoder

# Define input shape and latent dimension
height, width, channels = processed_levels.shape[1:4]
latent_dim = 32  # Example latent dimension
encoder, decoder = build_autoencoder((height, width, channels))

# Combine encoder and decoder into an autoencoder
autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(processed_levels, processed_levels, epochs=100, batch_size=32)

# Example: Sampling from the latent space to generate a new level
def generate_level(autoencoder, latent_dim):
    random_latent_vector = np.random.normal(size=(1, latent_dim))  # Sample random latent vector
    generated_level = decoder.predict(random_latent_vector)  # Decode to generate level
    return generated_level

generated_level = generate_level(autoencoder, latent_dim)

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
display_levels(processed_levels[0], np.clip(generated_level.squeeze(), 0, 1))  # Clip values for display
