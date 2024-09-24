# %%
import jax
import jumanji
import matplotlib.pyplot as plt
import numpy as np

# %%
# Instantiate a Jumanji environment using the registry
env = jumanji.make('Sokoban-v0')
# Reset your (jit-able) environment
key = jax.random.PRNGKey(0)
state, timestep = jax.jit(env.reset)(key)

# %%
# Run for 100 random steps
for step in range(100):
    # Render the current state
    env.render(state)

    # Generate a random action
    action = env.action_spec.generate_value()

    # Step the environment
    state, timestep = jax.jit(env.step)(state, action)

# Optionally, you could render the final state
env.render(state)
plt.title("Final Sokoban State after 100 Steps")
plt.show()
