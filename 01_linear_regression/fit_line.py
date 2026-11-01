import torch
from torch import nn

torch.manual_seed(42)

# Part A : Data
x = torch.linspace(-1, 1, 100).unsqueeze(1) # shape (100, 1)
y = 2 * x + 1 + 0.1 * torch.rand_like(x)    # shape (100, 1)

# Part B : Model, Loss, Optimizer
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("before:", model.weight.item(), model.bias.item())