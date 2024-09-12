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
memory = deque(maxlen=1000)  # <- replay buffer, deque - double-end queue
batch_size = 32
gamma = 0.99  # Discount factor
epsilon = 1.0
epsilon_decay = 0.995
learning_rate = 0.001
min_epsilon = 0.1

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



def train_step(params, batch, opt_state):
    def loss_fn(params):
        q_values = model.apply(params, batch["obs"])  # Current Q-values
        q_values = jnp.take_along_axis(q_values, batch["action"], axis=-1)  # Select Q-value for the action taken

         # Target Q-values
        next_q_values = model.apply(params, batch["next_obs"])
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
episodes = 100000  # Run 10000 iterations for training

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
            "action": jnp.expand_dims(jnp.array([entry.action for entry in batch]), axis=-1),
            "reward": jnp.array([entry.reward for entry in batch]),
            "next_obs": jnp.array([entry.next_obs for entry in batch]),
            "done": jnp.array([entry.done for entry in batch]),
        }
        params, opt_state = train_step(params, batch, opt_state)

    # Decay epsilon (epsilon-greedy exploration)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

# Observation
# env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")
# obs, info = env.reset()
# total_reward = 0  # Track total reward per episode
# test_episodes = 100  # Run 100 episodes with rendering

# for i in tqdm(range(test_episodes)):
#     rng, key = random.split(rng)
#     # Forward pass to get Q-values from the model
#     q_values = model.apply(params, jnp.array(obs).reshape(1, -1))
#     action = jnp.argmax(q_values).item()  # Greedy action (no exploration during testing)

#     # Step environment with selected action
#     next_obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     total_reward += reward  # Accumulate episode reward

#     # Reset environment if episode is done
#     if done:
#         print(f"Episode reward: {total_reward}")  # Track performance
#         obs, info = env.reset()  # Proper environment reset
#         total_reward = 0  # Reset reward for the new episode
#     else:
#         obs = next_obs

# # %%
