
# source: https://medium.com/@micky.multani/decoding-neural-networks-with-jax-from-untrained-to-trained-b148a48d87ff
# Import necessary libraries:
# jax and jax.numpy for numerical operations and JAX functionalities.
# flax.linen as nn, which is a neural network library for JAX.
# random and grad from JAX for initializing parameters and computing gradients.
# mnist from TensorFlow's Keras for loading the MNIST dataset.

#%%
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the dataset (this should already be done in your script)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize the first few images in the dataset
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(x_train[i], cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis('off')
# plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)  # Flatten

class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x
    
key = random.PRNGKey(0)
model = SimpleNN()
params = model.init(key, jnp.ones([1, 784]))['params']

def loss_fn(params, x, y):
    logits = model.apply({'params': params}, x)
    return -jnp.mean(jnp.sum(nn.log_softmax(logits) * jax.nn.one_hot(y, 10), axis=-1))

def accuracy(params, x, y):
    predictions = jnp.argmax(model.apply({'params': params}, x), axis=-1)
    return jnp.mean(predictions == y)

untrained_accuracy = accuracy(params, jnp.array(x_test), jnp.array(y_test))
print("Untrained model accuracy:", untrained_accuracy)

def predict(params, x):
    return jnp.argmax(model.apply({'params': params}, x), axis=-1)

# Predict on first few test images
untrained_predictions = predict(params, jnp.array(x_test[:9]))
print("Untrained model predictions:", untrained_predictions)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    # Reshape the image to 28x28 for display
    image = x_test[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {untrained_predictions[i]}, Actual: {y_test[i]}")
    plt.axis('off')
plt.suptitle("Untrained Model Predictions")
plt.show()

@jax.jit
def train_step(params, x, y, lr=0.001):
    grads = grad(loss_fn)(params, x, y)
    # Update each parameter by subtracting learning rate times the gradient
    updated_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return updated_params

for epoch in range(10):
    for i in range(0, len(x_train), 32):
        batch_x = jnp.array(x_train[i:i+32])
        batch_y = jnp.array(y_train[i:i+32])
        params = train_step(params, batch_x, batch_y)

print("Test accuracy:", accuracy(params, jnp.array(x_test), jnp.array(y_test)))

trained_predictions = predict(params, jnp.array(x_test[:9]))
print("Trained model predictions:", trained_predictions)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    # Reshape the image to 28x28 for display
    image = x_test[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {trained_predictions[i]}, Actual: {y_test[i]}")
    plt.axis('off')
plt.suptitle("Trained Model Predictions")
plt.show()

