import torch
from torch import nn

torch.manual_seed(42)

# Part A : Data
x = torch.linspace(-1, 1, 100).unsqueeze(1) # shape (100, 1)
y = 2 * x + 1 + 0.1 * torch.rand_like(x)    # shape (100, 1)

print(x[:10])
print(y[:10])