import jax
import optax
import os
from PIL import Image
import numpy as np
import jax.numpy as jnp
import glob
from tqdm import tqdm
from jax import random, jit
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

save_dir = './conv_visualizations'
os.makedirs(save_dir, exist_ok=True)


# dataset used: https://www.kaggle.com/datasets/koryakinp/fingers

def extract_label_from_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    label = basename[-2:]

    if label[0].isdigit() and label[1] in ['L', 'R']:
        fingers = int(label[0])
        hand_value = 0 if label[1] == 'L' else 1
        combined_label = fingers + (hand_value * 6)
        return combined_label
    else:
        raise ValueError(f"Unexpected label format in file: {filename}")

def load_images_from_folder(folder, fraction=0.4):
    images = []
    labels = []
    image_files = glob.glob(os.path.join(folder, '*.png'))
    
    if len(image_files) == 0:
        print(f"No images found in {folder}. Please check the directory.")
        return jnp.array(images), jnp.array(labels)

    subset_size = int(len(image_files) * fraction)

    for img_file in tqdm(image_files[:subset_size]):
        img = Image.open(img_file).convert('L')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)

        images.append(img_array)
        labels.append(extract_label_from_filename(img_file))

    return jnp.array(images), jnp.array(labels)

train_dir = './train'
test_dir = './test'

train_images, train_labels = load_images_from_folder(train_dir)
test_images, test_labels = load_images_from_folder(test_dir)

train_images = train_images[..., jnp.newaxis]
test_images = test_images[..., jnp.newaxis]

def init_cnn_params(rng):
    rng_conv1, rng_conv2, rng_dense1, rng_dense2 = random.split(rng, 4)
    params = {
        'conv1': random.normal(rng_conv1, (3, 3, 1, 32)) * np.sqrt(2.0 / (3 * 3 * 1)),
        'conv2': random.normal(rng_conv2, (3, 3, 32, 64)) * np.sqrt(2.0 / (3 * 3 * 32)),
        'dense1': random.normal(rng_dense1, (64 * 32 * 32, 128)) * np.sqrt(2.0 / (64 * 32 * 32)),
        'dense2': random.normal(rng_dense2, (128, 12)) * np.sqrt(2.0 / 128)
    }
    return params

def apply_dropout(rng, x, drop_rate=0.5):
    keep_prob = 1.0 - drop_rate
    mask = random.bernoulli(rng, keep_prob, x.shape)
    return jnp.where(mask, x / keep_prob, 0)

@jit
def cnn_forward(params, x, rng):
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    if x.ndim == 3:
        x = x[jnp.newaxis, ..., jnp.newaxis]
    elif x.ndim == 4 and x.shape[-1] != 1:
        x = x[..., jnp.newaxis]

    x = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv1'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))
    x = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv2'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))
    
    x = x.reshape((x.shape[0], -1))
    x = apply_dropout(rng, jax.nn.relu(jnp.dot(x, params['dense1'])))
    logits = jnp.dot(x, params['dense2'])
    return logits

def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=12)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))

def compute_grads(params, images, labels, rng):
    def loss_fn(params):
        logits = cnn_forward(params, images, rng)
        return cross_entropy_loss(logits, labels)

    return jax.value_and_grad(loss_fn)(params)

optimizer = optax.adam(1e-4)
params = init_cnn_params(random.PRNGKey(0))
opt_state = optimizer.init(params)

@jit
def update_params(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

@jit
def train_step(carry, rng):
    params, opt_state, train_images, train_labels = carry
    rng, dropout_rng = random.split(rng)
    loss, grads = compute_grads(params, train_images, train_labels, dropout_rng)
    params, opt_state = update_params(params, opt_state, grads)
    return (params, opt_state, train_images, train_labels), loss

def train_epoch(params, opt_state, rng, train_images, train_labels, num_epochs):
    carry = (params, opt_state, train_images, train_labels)
    rngs = random.split(rng, num_epochs)
    carry, losses = jax.lax.scan(train_step, carry, rngs)
    avg_loss = jnp.mean(losses)
    return carry[0], carry[1], avg_loss

rng = random.PRNGKey(42)
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    rng, input_rng = random.split(rng)
    params, opt_state, avg_loss = train_epoch(params, opt_state, input_rng, train_images, train_labels, num_epochs=1)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# ---------- visualizations
def cnn_forward_with_all_feature_maps(params, x, rng):
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    if x.ndim == 3:
        x = x[jnp.newaxis, ...]
    elif x.ndim == 2:
        x = x[jnp.newaxis, ..., jnp.newaxis]

    feature_maps_conv1 = jax.nn.relu(jax.lax.conv_general_dilated(x, params['conv1'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))
    feature_maps_conv2 = jax.nn.relu(jax.lax.conv_general_dilated(feature_maps_conv1, params['conv2'], (2, 2), 'SAME', dimension_numbers=dimension_numbers))

    x = feature_maps_conv2.reshape((feature_maps_conv2.shape[0], -1))
    x = apply_dropout(rng, jax.nn.relu(jnp.dot(x, params['dense1'])))
    logits = jnp.dot(x, params['dense2'])

    return logits, feature_maps_conv1, feature_maps_conv2

def plot_and_save_feature_maps(image, feature_maps_conv1, feature_maps_conv2, epoch, index, save_dir):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(image[..., 0], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(feature_maps_conv1[0, :, :, 0], cmap='gray')
    plt.title('Conv Layer 1 Output')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(feature_maps_conv2[0, :, :, 0], cmap='gray')
    plt.title('Conv Layer 2 Output')
    plt.axis('off')

    file_path = os.path.join(save_dir, f'epoch_{epoch}_image_{index}.png')
    plt.savefig(file_path)
    plt.close()

    print(f"Saved visualization for epoch {epoch}, image {index} at {file_path}")

def test_and_save_model(params, test_images, test_labels, rng, num_images=5, num_epochs=10, save_dir='./conv_visualizations'):
    random_indices = np.random.choice(len(test_images), num_images, replace=False)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} - Testing...")
        for idx, i in enumerate(random_indices):
            test_image = test_images[i]
            true_label = test_labels[i]

            logits, feature_maps_conv1, feature_maps_conv2 = cnn_forward_with_all_feature_maps(params, test_image, rng)

            plot_and_save_feature_maps(test_image, feature_maps_conv1, feature_maps_conv2, epoch + 1, idx, save_dir)

# test_and_save_model(params, test_images, test_labels, rng, num_images=5, num_epochs=10)

def animate_conv_layers_in_row(original_image, feature_maps_conv1, feature_maps_conv2, save_path1, save_path2, num_frames=50):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
    
    # Display the original image in the first column
    axes[0].imshow(original_image[..., 0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    ims_conv1 = []
    num_channels_conv1 = feature_maps_conv1.shape[-1]
    # Create an animation over all channels for Conv Layer 1
    for i in range(min(num_frames, num_channels_conv1)):
        im_conv1 = axes[1].imshow(feature_maps_conv1[0, :, :, i], cmap='viridis', animated=True)
        ims_conv1.append([im_conv1])
    axes[1].set_title("Conv Layer 1 Activation")
    axes[1].axis('off')
    
    ims_conv2 = []
    num_channels_conv2 = feature_maps_conv2.shape[-1]
    # Create an animation over all channels for Conv Layer 2
    for i in range(min(num_frames, num_channels_conv2)):
        im_conv2 = axes[2].imshow(feature_maps_conv2[0, :, :, i], cmap='viridis', animated=True)
        ims_conv2.append([im_conv2])
    axes[2].set_title("Conv Layer 2 Activation")
    axes[2].axis('off')
    
    ani_conv1 = animation.ArtistAnimation(fig, ims_conv1, interval=200, blit=True, repeat_delay=1000)
    ani_conv2 = animation.ArtistAnimation(fig, ims_conv2, interval=200, blit=True, repeat_delay=1000)
    
    ani_conv1.save(save_path1, writer='imagemagick')
    ani_conv2.save(save_path2, writer='imagemagick')
    
    plt.show()

random_index = np.random.randint(len(test_images))  # Randomly pick an image
logits, conv1_activations, conv2_activations = cnn_forward_with_all_feature_maps(params, test_images[random_index], rng)

# Animate and save the convolutional layers side by side
animate_conv_layers_in_row(test_images[random_index], conv1_activations, conv2_activations, 
                           save_path1='conv1_animation_with_image_row.gif', 
                           save_path2='conv2_animation_with_image_row.gif')