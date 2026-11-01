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
losses = []
for epoch in range(1,201):
    pred = model(x)                # 1. forward
    loss = loss_fn(pred, y)        # 2. loss
    losses.append(loss.item())  
    optimizer.zero_grad()          # 3. clear old gradients
    loss.backward()                # 4. compute gradients
    optimizer.step()               # 5. update parameters

    if epoch%20==0:
        print(f"{epoch:3d} | loss {loss.item():.5f} | "
              f"w {model.weight.item():.4f} | b {model.bias.item():.4f}")

print("after:", model.weight.item(), model.bias.item())

# Part D : Prediction
model.eval()

with torch.no_grad():
    x_new = torch.tensor([[0.0], [0.5], [2.0]]) # shape (3, 1)
    y_new = model(x_new)

for xi, yi in zip(x_new, y_new):
    print(f"x = {xi.item():5.2f} -> predicted {yi.item():.3f} | true {2 * xi.item() + 1:.3f}")