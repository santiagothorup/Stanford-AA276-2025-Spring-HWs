from homework2_problem4_helper import *
from part1 import state_limits
import numpy as np
import torch

path = 'outputs/cbf.ckpt'
neuralcbf = NeuralCBF(path)

upper_bound , lower_bound = state_limits()
total_vol = np.prod(upper_bound - lower_bound)

sample_size = 1000
count = 0
for i in range(sample_size):
    x = np.random.uniform(lower_bound, upper_bound, (1, 13))
    x = torch.tensor(x, dtype=torch.float32)
    h_value = neuralcbf.values(x).item()
    if h_value > 0:
        count += 1

safe_ratio = count / sample_size
safe_volume = safe_ratio * total_vol

print(f"Safe volume: {safe_volume:.4f}")
print(f"Safe ratio: {safe_ratio:.4f}")