from problem4_helper import *
import numpy as np
import torch
import sys
sys.path.append('../hw1')  # Add path to hw1 directory if needed
from part1 import state_limits  # Import state_limits function

# Initialize the Neural Value Function with the checkpoint
path = 'outputs/vf.ckpt'
neuralvf = NeuralVF(path)

# Get state space bounds
upper_bound, lower_bound = state_limits()
upper_np = upper_bound.numpy()
lower_np = lower_bound.numpy()

# Calculate total volume of the sampling region
total_vol = np.prod(upper_np - lower_np)

# Parameters for sampling
sample_size = 10000
count = 0

for i in range(sample_size):
    # Generate a random state within bounds - one dimension at a time
    x_np = np.zeros((1, 13))
    for j in range(13):
        x_np[0, j] = np.random.uniform(lower_np[j], upper_np[j])
    
    # Convert to PyTorch tensor for evaluation
    x = torch.tensor(x_np, dtype=torch.float32)
    
    # Evaluate value function at this state
    # For value functions, the safe set is typically where V(x) ≤ 0
    vf_value = neuralvf.values(x).item()
    
    # Check if state is in safe set (V(x) ≤ 0)
    if vf_value <= 0:
        count += 1

# Calculate volume
safe_ratio = count / sample_size
safe_volume = safe_ratio * total_vol

print(f"Total samples: {sample_size}")
print(f"Safe samples: {count}")
print(f"Safe ratio: {safe_ratio:.4f}")
print(f"Total volume: {total_vol:.4e}")
print(f"Safe volume: {safe_volume:.4e}")