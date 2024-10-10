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
def load_images_from_folder(folder, fraction=0.3):
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
train_dir = './train'
test_dir = './test'

print('Loading train images...')
train_images, train_labels = load_images_from_folder(train_dir)

print('Loading test images...')
test_images, test_labels = load_images_from_folder(test_dir)

print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")
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
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

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
    one_hot_labels = jax.nn.one_hot(labels, num_classes=7)
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

    # Ensure images have a channel dimension
    images, labels = train_images, train_labels
    images = images[..., jnp.newaxis]

    # Compute loss and gradients
    loss, grads = compute_grads(params, images, labels, dropout_rng)

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
num_epochs = 30
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
