import jax
import optax
import os
from PIL import Image
import numpy as np
import jax.numpy as jnp
import glob
from tqdm import tqdm
from jax import random, jit, grad
import jax.lax
import matplotlib.pyplot as plt

save_dir = './conv_visualizations'
os.makedirs(save_dir, exist_ok=True) 

def visualize_and_save_feature_maps(params, image, rng, epoch, save_dir):
    logits, feature_maps_conv1 = cnn_forward_with_feature_maps(params, image, rng)

    feature_maps_conv2 = jax.nn.relu(jax.lax.conv_general_dilated(
        feature_maps_conv1, params['conv2'], (2, 2), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')))

    # Create a new figure for the visualization
    plt.figure(figsize=(15, 10))

    # Plot the original image
    plt.subplot(3, 1, 1)
    plt.imshow(image[..., 0], cmap='gray')
    plt.title(f'Original Image - Epoch {epoch}')
    plt.axis('off')

    # Plot the first conv layer feature maps (conv1)
    num_channels_conv1 = feature_maps_conv1.shape[-1]
    num_columns = 8
    num_rows = (num_channels_conv1 + num_columns - 1) // num_columns  # Calculate rows based on columns
    for i in range(num_channels_conv1):
        plt.subplot(3, num_columns, num_columns + i + 1)
        plt.imshow(feature_maps_conv1[0, :, :, i], cmap='gray')
        plt.title(f'conv1 - channel {i}')
        plt.axis('off')

    # Plot the second conv layer feature maps (conv2)
    num_channels_conv2 = feature_maps_conv2.shape[-1]
    for i in range(num_channels_conv2):
        plt.subplot(3, num_columns, 2 * num_columns + i + 1)
        plt.imshow(feature_maps_conv2[0, :, :, i], cmap='gray')
        plt.title(f'conv2 - channel {i}')
        plt.axis('off')

    # Save the figure
    file_path = os.path.join(save_dir, f'epoch_{epoch}.png')
    plt.savefig(file_path)
    plt.close()

    print(f"Saved visualization for epoch {epoch} at {file_path}")

# Function to extract label from the file name based on the last two characters
def extract_label_from_filename(filename):
    # Remove the file extension and then extract the last two characters
    basename = os.path.splitext(os.path.basename(filename))[0]  # Remove the path and extension
    # print(f"Extracting label from basename: {basename}")  # Debug print

    label = basename[-2:]  # Extracts the last two characters like '5L' or '2R'
    # print(f"Extracted label: {label}")  # Debug print

    # Ensure the label follows the expected pattern
    if label[0].isdigit() and label[1] in ['L', 'R']:
        fingers = int(label[0])  # 0-5 indicates the number of fingers
        hand = label[1]  # 'L' for left, 'R' for right

        # Encoding: Left = 0, Right = 1. Combine with the number of fingers
        hand_value = 0 if hand == 'L' else 1
        combined_label = fingers + (hand_value * 6)

        return combined_label
    else:
        raise ValueError(f"Unexpected label format in file: {filename}")

# Updated function to load images and extract labels from filenames
def load_images_from_folder(folder, fraction=0.3):
    images = []
    labels = []
    image_files = glob.glob(os.path.join(folder, '*.png'))  # Match all PNG files

    print(f"Number of images found in {folder}: {len(image_files)}")

    if len(image_files) == 0:
        print(f"No images found in {folder}. Please check the directory and file format.")
        return jnp.array(images), jnp.array(labels)

    subset_size = int(len(image_files) * fraction)

    for img_file in tqdm(image_files[:subset_size]):
        img = Image.open(img_file).convert('L')
        img = img.resize((128, 128))  # Resize to 128x128 for Fingers dataset
        img_array = np.array(img) / 255.0
        images.append(img_array)

        # print(f"file {img_file}")  # Print file for debugging

        # Extract label from the filename
        label = extract_label_from_filename(img_file)
        labels.append(label)

    return jnp.array(images), jnp.array(labels)

# Directories for train and test
train_dir = './train'
test_dir = './test'

print('Loading train images...')
train_images, train_labels = load_images_from_folder(train_dir)

print('Loading test images...')
test_images, test_labels = load_images_from_folder(test_dir)
train_images = train_images[..., jnp.newaxis]  # Add channel dimension
test_images = test_images[..., jnp.newaxis]
print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")

# Initialize CNN parameters for 128x128 input images
def init_cnn_params(rng):
    rng_conv1, rng_conv2, rng_dense1, rng_dense2 = random.split(rng, 4)
    params = {
        'conv1': random.normal(rng_conv1, (3, 3, 1, 32)),  # Conv1 expects 1 input channel for grayscale
        'conv2': random.normal(rng_conv2, (3, 3, 32, 64)),  # Conv2 expects 32 channels
        'dense1': random.normal(rng_dense1, (64 * 32 * 32, 128)),  # Updated for 128x128 input size
        'dense2': random.normal(rng_dense2, (128, 12))  # Output layer for 12 classes (0-5 for left, 6-11 for right)
    }
    return params

# Apply dropout
def apply_dropout(rng, x, drop_rate=0.5):
    keep_prob = 1.0 - drop_rate
    mask = random.bernoulli(rng, keep_prob, x.shape)
    return jnp.where(mask, x / keep_prob, 0)

# CNN forward pass with dropout
@jit
def cnn_forward(params, x, rng):
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    # Ensure input x has the shape (batch_size, height, width, channels)
    if x.ndim == 3:  # If x has no batch dimension, add it
        x = x[jnp.newaxis, ..., jnp.newaxis]  # Add batch and channel dimensions
    elif x.ndim == 4 and x.shape[-1] != 1:  # If x has batch dimension but no channel, add the channel dimension
        x = x[..., jnp.newaxis]

    # Apply conv1
    x = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv1'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))

    # Apply conv2
    x = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv2'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))

    # Flatten the feature map and apply the dense layers
    x = x.reshape((x.shape[0], -1))
    x = apply_dropout(rng, jax.nn.relu(jnp.dot(x, params['dense1'])))
    logits = jnp.dot(x, params['dense2'])
    return logits
# Cross entropy loss
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=12)  # Updated to 12 classes
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))

# Compute gradients using vmap
def compute_grads(params, images, labels, rng):
    def loss_fn(params):
        logits = cnn_forward(params, images, rng)
        loss = cross_entropy_loss(logits, labels)
        return loss

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# Update optimizer parameters
optimizer = optax.adam(1e-3)
params = init_cnn_params(random.PRNGKey(0))
opt_state = optimizer.init(params)

# Update function
@jit
def update_params(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

@jit
def train_step(carry, rng):
    params, opt_state, train_images, train_labels = carry  # Unpack only 4 values
    rng, dropout_rng = random.split(rng)  # Split the rng for dropout

    # Ensure train_images has batch and channel dimensions
    if train_images.ndim == 3:  # If the images don't have batch dimension
        train_images = train_images[jnp.newaxis, ..., jnp.newaxis]
    elif train_images.ndim == 4 and train_images.shape[-1] != 1:  # If no channel dimension
        train_images = train_images[..., jnp.newaxis]

    # Compute loss and gradients
    loss, grads = compute_grads(params, train_images, train_labels, dropout_rng)

    # Update parameters
    params, opt_state = update_params(params, opt_state, grads)

    return (params, opt_state, train_images, train_labels), loss

# Training epoch with scan and accumulation of loss
def train_epoch(params, opt_state, rng, train_images, train_labels, num_epochs):
    carry = (params, opt_state, train_images, train_labels)  # Initialize carry with 4 values
    rngs = random.split(rng, num_epochs)  # Generate separate RNG for each epoch
    carry, losses = jax.lax.scan(train_step, carry, rngs)  # Pass rngs as xs to train_step
    avg_loss = jnp.mean(losses)  # Compute average loss over all epochs
    return carry[0], carry[1], avg_loss


# Example training loop using lax.scan
rng = random.PRNGKey(42)
num_epochs = 20
for epoch in range(num_epochs):
    rng, input_rng = random.split(rng)
    params, opt_state, avg_loss = train_epoch(params, opt_state, input_rng, train_images, train_labels, num_epochs=1)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# Testing function for classifying random test images
def classify_and_display_image(params, test_image, true_label):
    test_image = test_image[jnp.newaxis, ..., jnp.newaxis]
    logits = cnn_forward(params, test_image, rng)
    predicted_label = jnp.argmax(logits, axis=-1)
    plt.imshow(test_image[0, ..., 0], cmap='gray')
    plt.title(f"True label: {true_label}, Predicted label: {predicted_label[0]}")
    plt.axis('off')
    plt.show()

def test_model(params, test_images, test_labels, num_images=5):
    random_indices = np.random.choice(len(test_images), num_images, replace=False)
    for i in random_indices:
        classify_and_display_image(params, test_images[i], test_labels[i])

test_model(params, test_images, test_labels)
