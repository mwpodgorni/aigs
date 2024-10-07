# lax.map(partial(apply_kernel, x[0]), kernels) # -> what you want to plot


import jax
import optax
import os
from PIL import Image
import numpy as np
import jax.numpy as jnp
import glob
from tqdm import tqdm
from jax import random, jit, grad, vmap
import jax.lax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class_mapping = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

# Function to load images from a directory and map subfolder names to labels
def load_images_from_folder(folder, fraction=0.7):
    images = []
    labels = []
    for emotion, label in tqdm(class_mapping.items()):
        emotion_folder = os.path.join(folder, emotion)
        if os.path.isdir(emotion_folder):
            image_files = glob.glob(os.path.join(emotion_folder, '*.jpg'))
            subset_size = int(len(image_files) * fraction)
            for img_file in image_files[:subset_size]:
                img = Image.open(img_file).convert('L')
                img = img.resize((48, 48))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(label)
    return jnp.array(images), jnp.array(labels)


# Load train and test datasets
train_dir = './train'
test_dir = './test'

print('load train images')
train_images, train_labels = load_images_from_folder(train_dir)
print('load test images')
test_images, test_labels = load_images_from_folder(test_dir)

print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")


# Load train and test datasets
train_dir = './train'
test_dir = './test'
train_images, train_labels = load_images_from_folder(train_dir)
test_images, test_labels = load_images_from_folder(test_dir)

# Initialize CNN parameters
def init_cnn_params(rng):
    rng_conv1, rng_conv2, rng_dense1, rng_dense2 = random.split(rng, 4)
    params = {
        'conv1': random.normal(rng_conv1, (3, 3, 1, 32)),  # Conv1 expects 1 input channel for grayscale
        'conv2': random.normal(rng_conv2, (3, 3, 32, 64)),  # Conv2 expects 32 channels
        'dense1': random.normal(rng_dense1, (64 * 12 * 12, 128)),  # Dense1 layer
        'dense2': random.normal(rng_dense2, (128, 7))  # Output layer (7 classes)
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
    # Explicitly define the dimension numbers: (NHWC for input, HWIO for filter, NHWC for output)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    
    # Apply conv1 with explicit dimension_numbers
    print(f"Shape before conv1: {x.shape}")  # Check shape before first conv
    x = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv1'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))
    print(f"Shape after conv1: {x.shape}")  # Check shape after first conv
    
    # Apply conv2 with explicit dimension_numbers
    x = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv2'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))
    print(f"Shape after conv2: {x.shape}")  # Check shape after second conv
    
    # Flatten the feature map and apply the dense layers
    x = x.reshape((x.shape[0], -1))  # Flatten the feature map
    x = apply_dropout(rng, jax.nn.relu(jnp.dot(x, params['dense1'])))  # Apply dropout
    logits = jnp.dot(x, params['dense2'])  # Final dense layer
    return logits

# Cross entropy loss
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=7)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))

# Compute gradients using vmap
def compute_grads(params, images, labels, rng):
    def loss_fn(params):
        logits = cnn_forward(params, images, rng)  # Forward pass
        loss = cross_entropy_loss(logits, labels)  # Calculate loss
        return loss
    
    # Compute gradients with respect to the parameters
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# Update optimizer parameters
optimizer = optax.adam(1e-3)
params = init_cnn_params(random.PRNGKey(0))
opt_state = optimizer.init(params)

# Update function
@jit
def update_params(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state, params)  # Ensure updates are applied with respect to params
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Training step using jax.lax.scan
@jit
def train_step(carry, images, labels):
    params, opt_state, rng, _ = carry
    print(f"Image shape before adding channel: {images.shape}")  # Check input shape before adding channel
    images = images[..., jnp.newaxis]  # Add channel dimension for grayscale images
    print(f"Image shape after adding channel: {images.shape}")  # Check input shape after adding channel
    loss, grads = compute_grads(params, images, labels, rng)
    params, opt_state = update_params(params, opt_state, grads)
    return (params, opt_state, rng, loss), loss

# Training epoch with scan
def train_epoch(params, opt_state, rng, train_images, train_labels):
    carry = (params, opt_state, rng, 0)
    carry, loss = train_step(carry, train_images, train_labels)  # Pass the entire dataset to train_step
    return carry[0], carry[1], loss

# Helper to get mini-batches
def get_batch(images, labels, batch_size):
    idx = np.random.choice(len(images), batch_size, replace=False)
    return images[idx], labels[idx]

# Plot and animate convolutional filters
def plot_conv_filters(params, epoch):
    conv1_filters = params['conv1']
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(conv1_filters[:, :, 0, i], cmap='gray')
        ax.axis('off')
    plt.savefig(f'conv_filters_epoch_{epoch}.png')

# Example training loop
rng = random.PRNGKey(42)
num_epochs = 50
print('training')
for epoch in tqdm(range(num_epochs)):
    print(f"Epoch {epoch + 1}")
    params, opt_state, loss = train_epoch(params, opt_state, rng, train_images, train_labels)
    print(f"Loss: {loss}")
    # plot_conv_filters(params, epoch)  # Save filter images for each epoch

def classify_and_display_image(params, test_image, true_label):
    # Add the channel dimension for the test image
    test_image = test_image[jnp.newaxis, ..., jnp.newaxis]  # Add batch and channel dimension (1 image, grayscale)
    
    # Run the CNN forward pass to get logits
    logits = cnn_forward(params, test_image, rng)
    
    # Get the predicted label (argmax of logits)
    predicted_label = jnp.argmax(logits, axis=-1)
    
    # Display the image and the predicted label
    plt.imshow(test_image[0, ..., 0], cmap='gray')  # Remove batch and channel dimension for display
    plt.title(f"True label: {true_label}, Predicted label: {predicted_label[0]}")
    plt.axis('off')
    plt.show()

# Example of testing two images
def test_model(params, test_images, test_labels, num_images=5):
    # Select two test images to classify
    random_indices = np.random.choice(len(test_images), num_images, replace=False)
    for i in random_indices:
        classify_and_display_image(params, test_images[i], test_labels[i])

print("Testing the model with two test images:")
test_model(params, test_images, test_labels)