# %% Imports (add as needed) #############################################
import gymnasium as gym
from jax import random
from tqdm import tqdm
from collections import deque, namedtuple
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
import numpy as np  # Needed for sampling from replay buffer
import random as py_random  # For Python random sampling

# %% Constants ###########################################################
print('1 constants')
env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=10000)  # <- replay buffer, deque - double-end queue
batch_size = 63
gamma = 0.99  # Discount factor
epsilon = 1.0
epsilon_decay = 0.99
learning_rate = 0.001
min_epsilon = 0.1
target_update_freq = 1000  # Frequency to update target network

# %% Model ###############################################################
print('2 model')
class DQNModel(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)  # hidden layer 1
        x = nn.relu(x)
        x = nn.Dense(128)(x)  # hidden layer 2
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)  # output layer (Q-values for actions)
        return x

# Initialize model
action_dim = env.action_space.n
model = DQNModel(action_dim=action_dim)

key, rng = random.split(rng)
print("Observation space:", env.observation_space)
print("Observation space shape:", env.observation_space.shape)
params = model.init(key, jnp.ones((1, env.observation_space.shape[0])))

# Initialize target network with the same parameters
target_params = params

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

def select_action(rng, q_values, epsilon):
    rng, subkey = random.split(rng)
    # Random or Greedy action based on epsilon-greedy strategy
    if random.uniform(subkey, (), minval=0.0, maxval=1.0) < epsilon:
        rng, subkey = random.split(rng)
        return random.randint(subkey, (1,), 0, action_dim).item()  # Random action
    else:
        return jnp.argmax(q_values).item()  # Greedy action


# Modify train_step to use target_params for next_q_values
def train_step(params, target_params, batch, opt_state):
    def loss_fn(params):
        q_values = model.apply(params, batch["obs"])  # Current Q-values, shape (batch_size, action_dim)
        q_values = jnp.take_along_axis(q_values, batch["action"], axis=-1)  # Q-value for the action taken, shape (batch_size, 1)
        q_values = jnp.squeeze(q_values, axis=-1)  # Shape (batch_size,)

        # Target Q-values from target network
        next_q_values = model.apply(target_params, batch["next_obs"])  # Shape (batch_size, action_dim)
        target_q_values = batch["reward"] + (1.0 - batch["done"]) * gamma * jnp.max(next_q_values, axis=-1)

        return jnp.mean((q_values - target_q_values) ** 2)

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# %% Environment #########################################################
print('3 environment')
# observation
obs, info = env.reset()
total_reward = 0.0  # Track total reward per episode
episodes = 100000  # Run 100000 iterations for training
step_counter = 0  # Step counter for target network update

for i in tqdm(range(episodes)):
    rng, key = random.split(rng)
    # Forward pass to get Q-values from the model
    q_values = model.apply(params, jnp.array(obs).reshape(1, -1))
    action = select_action(key, q_values, epsilon)  # Select action

    # Step environment with selected action
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    memory.append(entry(obs, action, reward, next_obs, done))  # Store transition in replay buffer

    total_reward += reward  # Accumulate episode reward
    # Reset environment if episode is done
    if done:
        print(f"Episode reward: {total_reward}")  # Track performance
        obs, info = env.reset()  # Proper environment reset
        total_reward = 0  # Reset reward for the new episode
    else:
        obs = next_obs

    # Perform a training step if there's enough data in the buffer
    if len(memory) > batch_size:
        batch = py_random.sample(memory, batch_size)
        batch = {
            "obs": jnp.array([entry.obs for entry in batch]),
            "action": jnp.array([entry.action for entry in batch]).reshape(-1, 1),  # Shape (batch_size, 1)
            "reward": jnp.array([entry.reward for entry in batch]),
            "next_obs": jnp.array([entry.next_obs for entry in batch]),
            "done": jnp.array([entry.done for entry in batch], dtype=jnp.float32),
        }
        params, opt_state = train_step(params, target_params, batch, opt_state)

    # Update the target network periodically
    if step_counter % target_update_freq == 0:
        target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

    step_counter += 1

    # Decay epsilon (epsilon-greedy exploration)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()
