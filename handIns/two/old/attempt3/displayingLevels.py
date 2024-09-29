import jax
import jumanji
import matplotlib.pyplot as plt

# Initialize RNG and environment
rng = jax.random.PRNGKey(0)
env = jumanji.make("Sokoban-v0")

# Function to render the environment
def render_level(state):
    # Render the environment directly
    env.render(state)
    
    # Display a title for the rendered level
    plt.title("Sokoban Level (rendered directly)")
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Load and display 5 levels one after another
def display_multiple_levels(num_levels):
    for i in range(num_levels):
        # Split the random key to get a new one for each level
        global rng  # Use the global rng variable
        rng, subkey = jax.random.split(rng)

        # Reset the environment to get the initial state
        state, timestep = env.reset(subkey)

        # Render the level
        render_level(state)

# Display 5 different levels one after another
display_multiple_levels(5)

