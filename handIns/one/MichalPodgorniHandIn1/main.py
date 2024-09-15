# %% Imports #############################################
import gymnasium as gym
from jax import random
from tqdm import tqdm
from collections import deque, namedtuple
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
import numpy as np
import random as py_random 
import pickle
import os
# %% Constants ###########################################################
SAVE_PATH = "dqn_training_state.pkl" 
env = gym.make("CartPole-v1", render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=10000)
batch_size = 32
gamma = 0.99  # Discount factor
epsilon = 1.0
epsilon_decay = 0.995
learning_rate = 0.0003
min_epsilon = 0.1
target_update_freq = 50

# %% Model ###############################################################
class DQNModel(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)  # hidden layer 1
        x = nn.relu(x)
        x = nn.Dense(128)(x)  # hidden layer 2
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)  # output layer
        return x

# initialize model
action_dim = env.action_space.n
model = DQNModel(action_dim=action_dim)

key, rng = random.split(rng)
params = model.init(key, jnp.ones((1, env.observation_space.shape[0])))

# initialize target network with the same parameters
target_params = params

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Function to save training state
def save_state(episode, params, target_params, opt_state, memory, epsilon, step_counter):
    memory_serializable = [{
        "obs": entry.obs.tolist(),
        "action": entry.action,
        "reward": entry.reward,
        "next_obs": entry.next_obs.tolist(),
        "done": entry.done
    } for entry in memory]
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({
            "episode": episode,
            "params": params,
            "target_params": target_params,
            "opt_state": opt_state,
            "memory": memory_serializable,
            "epsilon": epsilon,
            "step_counter": step_counter
        }, f)
    print(f"Training state saved at episode {episode}.")

def load_state():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            state = pickle.load(f)
            print(f"Resuming training from episode {state['episode']}.")
            
            memory = deque([
                entry(
                    obs=np.array(item["obs"]),
                    action=item["action"],
                    reward=item["reward"],
                    next_obs=np.array(item["next_obs"]),
                    done=item["done"]
                ) for item in state["memory"]
            ], maxlen=10000)
            
            return {
                "episode": state["episode"],
                "params": state["params"],
                "target_params": state["target_params"],
                "opt_state": state["opt_state"],
                "memory": memory,
                "epsilon": state["epsilon"],
                "step_counter": state["step_counter"]
            }
    return None

def random_policy_fn(rng, q_values, epsilon):
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def select_action(rng, q_values, epsilon):
    rng, subkey = random.split(rng)
    # random or greedy action based on epsilon-greedy strategy
    if random.uniform(subkey, (), minval=0.0, maxval=1.0) < epsilon:
        rng, subkey = random.split(rng)
        return random.randint(subkey, (1,), 0, action_dim).item() 
    else:
        return jnp.argmax(q_values).item() 

def train_step(params, target_params, batch, opt_state):
    # mean square error loss function
    def loss_fn(params):
        q_values = model.apply(params, batch["obs"])
        q_values = jnp.take_along_axis(q_values, batch["action"], axis=-1) 
        q_values = jnp.squeeze(q_values, axis=-1)

        # target q-values from target network
        next_q_values = model.apply(target_params, batch["next_obs"])
        target_q_values = batch["reward"] + (1.0 - batch["done"]) * gamma * jnp.max(next_q_values, axis=-1)

        return jnp.mean((q_values - target_q_values) ** 2)

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# %% Environment #########################################################
state = load_state()
if state:
    episode_start = state["episode"] + 1 
    params = state["params"]
    target_params = state["target_params"]
    opt_state = state["opt_state"]
    memory = state["memory"]
    epsilon = state["epsilon"]
    step_counter = state["step_counter"]
else:
    episode_start = 1
    step_counter = 0
    memory = deque(maxlen=10000)

obs, info = env.reset()
total_reward = 0.0
episodes = 100
step_counter = 0
sum_reward = 0
for i in tqdm(range(episodes)):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
            rng, key = random.split(rng)
            # forward pass to get Q-values from the model
            q_values = model.apply(params, jnp.array(obs).reshape(1, -1))
            action = select_action(key, q_values, epsilon)

            # step environment with selected action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            memory.append(entry(obs, action, reward, next_obs, done))

            total_reward += reward
            obs = next_obs

            # training step
            if len(memory) > batch_size:
                batch = py_random.sample(memory, batch_size)
                batch = {
                    "obs": jnp.array([entry.obs for entry in batch]),
                    "action": jnp.array([entry.action for entry in batch]).reshape(-1, 1),
                    "reward": jnp.array([entry.reward for entry in batch]),
                    "next_obs": jnp.array([entry.next_obs for entry in batch]),
                    "done": jnp.array([entry.done for entry in batch], dtype=jnp.float32),
                }
                params, opt_state = train_step(params, target_params, batch, opt_state)

            # update target network periodically
            if step_counter % target_update_freq == 0:
                target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

            step_counter += 1

            # decay epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

    sum_reward += total_reward
    print(f"Episode {i+1} reward: {total_reward}")
    if i % 50 == 0:
        print('saving state')
        save_state(i, params, target_params, opt_state, memory, epsilon, step_counter)
print('Average reward:', sum_reward/episodes)

env.close()
