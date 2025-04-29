from homework2_problem4_helper import *
from part1 import state_limits
import numpy as np
import torch

path = 'outputs/cbf.ckpt'
neuralcbf = NeuralCBF(path)

upper_bound , lower_bound = state_limits()
upper_np = upper_bound.numpy()
lower_np = lower_bound.numpy()

total_vol = np.prod(upper_np - lower_np)

sample_size = 1000
count = 0
for i in range(sample_size):
    # Generate a random state within bounds - one dimension at a time
    x_np = np.zeros((1, 13))
    for j in range(13):
        x_np[0, j] = np.random.uniform(lower_np[j], upper_np[j])
    
    # Convert to PyTorch tensor for the CBF evaluation
    x = torch.tensor(x_np, dtype=torch.float32)
    
    # Evaluate CBF at this state
    h_value = neuralcbf.values(x).item()
    
    # Check if state is in safe set (h(x) ≥ 0)
    if h_value >= 0:
        count += 1

safe_ratio = count / sample_size
safe_volume = safe_ratio * total_vol

print(f"Total samples: {sample_size}")
print(f"Safe samples: {count}")
print(f"Safe ratio: {safe_ratio:.4f}")
print(f"Total volume: {total_vol:.4e}")
print(f"Safe volume: {safe_volume:.4e}")