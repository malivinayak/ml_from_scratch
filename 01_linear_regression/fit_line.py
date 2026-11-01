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

# Part C : Training Loop
for epoch in range(1,201):
    pred = model(x)                # 1. forward
    loss = loss_fn(pred, y)        # 2. loss
    optimizer.zero_grad()          # 3. clear old gradients
    loss.backward()                # 4. compute gradients
    optimizer.step()               # 5. update parameters

    if epoch%20==0:
        print(f"{epoch:3d} | loss {loss.item():.5f} | "
              f"w {model.weight.item():.4f} | b {model.bias.item():.4f}")


print("after:", model.weight.item(), model.bias.item())
