# %% import
import gym
import gym_sokoban
import numpy as np
import matplotlib.pyplot as plt
import optax
import pickle
# %% Get levels -----------------------------------
# Create Sokoban environment
env = gym.make('Sokoban-v0')
# Initialize levels
levels_per_file = 25
for i in range(4):
    # Initialize levels for this section
    levels = []
    
    # Collect 250 levels
    for l in range(levels_per_file):
        observation = env.reset()
        print('-------')
        print(observation)
        print('-------')
        levels.append(observation)
        if l % 10 == 0:
            print(f"Still going... ,", l)
    
    # Convert levels to a numpy array
    levels_array = np.array(levels)
    
    # Save levels to a pickle file
    with open(f'sokoban_levels_part_{i+1}.pkl', 'wb') as f:
        pickle.dump(levels_array, f)
    
    print(f"Levels saved to sokoban_levels_part_{i+1}.pkl.")
# %% Combine levels
all_levels = []
# Load each pickle file and append levels to all_levels
for i in range(4):
    with open(f'sokoban_levels_part_{i+1}.pkl', 'rb') as f:
        levels = pickle.load(f)
        all_levels.extend(levels)
# Convert all levels to a numpy array
all_levels_array = np.array(all_levels)
# Save the combined levels to a new pickle file
with open('sokoban_levels_combined.pkl', 'wb') as f:
    pickle.dump(all_levels_array, f)
print("All levels combined and saved to sokoban_levels_combined.pkl.")
# %% Show level ---------------------------------
with open('sokoban_levels_combined.pkl', 'rb') as f:
    all_levels_array = pickle.load(f)
# Show one level, for example, the first one
level_index = 9
plt.imshow(all_levels_array[level_index])
plt.title("Sokoban Level")
plt.axis('off')
plt.show()
# %%