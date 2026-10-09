"""
ARIA-ICNABS Project
File: 01_Train_PINN_ABC_L1.py
Purpose: PINN with GENUINE ABC-L1 discretization (Mittag-Leffler kernel).
         This is the truly nonlocal, nonsingular-kernel residual.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    L = 1.0
    T = 1.0
    D0 = 0.01
    beta = 0.5
    v = 0.1
    alpha = 0.8
    C0 = 0.0
    Cin = 1.0

    # Network
    n_hidden = 4
    n_neurons = 64

    # Training
    n_epochs = 8000
    lr = 1e-3
    n_ic = 300
    n_bc = 300

    # ABC-L1 time grid
    n_time_steps = 50       # smaller for speed
    n_x_colloc = 60

config = Config()

# ============================================================
# PINN NETWORK
# ============================================================
class PINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [nn.Linear(2, config.n_neurons), nn.Tanh()]
        for _ in range(config.n_hidden - 1):
            layers += [nn.Linear(config.n_neurons, config.n_neurons), nn.Tanh()]
        layers.append(nn.Linear(config.n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# ============================================================
# LOAD ABC-L1 WEIGHTS
# ============================================================
def load_abc_weights(alpha, dt, Nt):
    cache = f"abc_weights_alpha{alpha:.3f}_Nt{Nt}.npz"
    if os.path.exists(cache):
        print(f"Loading cached ABC weights: {cache}")
        return torch.tensor(np.load(cache)['W'], dtype=torch.float32)

    print(f"Precomputing ABC weights: alpha={alpha}, Nt={Nt}")
    from abc_l1_kernel import precompute_abc_weight_matrix
    W = precompute_abc_weight_matrix(alpha, dt, Nt, verbose=True)
    np.savez(cache, W=W, alpha=alpha, dt=dt, Nt=Nt)
    return torch.tensor(W, dtype=torch.float32)

# ============================================================
# LOSS
# ============================================================
def compute_loss(model, config, W):
    Nt = config.n_time_steps
    dt = config.T / Nt

    # ---- PDE residual via ABC-L1 ----
    x_pde = torch.rand(config.n_x_colloc, 1) * config.L
    time_grid = torch.linspace(0, config.T, Nt + 1)

    # Forward: C(x, t_k) for k = 0..Nt
    history_C = []
    for k in range(Nt + 1):
        t_k = time_grid[k].reshape(1, 1).expand(config.n_x_colloc, 1)
        history_C.append(model(x_pde, t_k))

    # Residual sum
    loss_pde = 0.0
    for n in range(1, Nt + 1):
        # ABC-L1: D_t^alpha C(x, t_n) ≈ sum_k W[n,k] * (C_{k+1} - C_k)
        frac = 0.0
        for k in range(n):
            frac = frac + W[n, k] * (history_C[k + 1] - history_C[k])

        # Spatial derivatives via autograd
        x_ad = x_pde.clone().requires_grad_(True)
        t_n = time_grid[n].reshape(1, 1).expand(config.n_x_colloc, 1)
        t_ad = t_n.clone().requires_grad_(True)
        C_ad = model(x_ad, t_ad)

        dC_dx = torch.autograd.grad(
            C_ad, x_ad, torch.ones_like(C_ad), create_graph=True
        )[0]
        d2C_dx2 = torch.autograd.grad(
            dC_dx, x_ad, torch.ones_like(dC_dx), create_graph=True
        )[0]

        D_x = config.D0 * (1 + config.beta * x_ad)
        dD_dx = config.D0 * config.beta
        diffusion = dD_dx * dC_dx + D_x * d2C_dx2
        advection = config.v * dC_dx

        residual = frac - diffusion + advection
        loss_pde = loss_pde + torch.mean(residual**2)

    loss_pde = loss_pde / Nt

    # ---- IC ----
    x_ic = torch.rand(config.n_ic, 1) * config.L
    t_ic = torch.zeros(config.n_ic, 1)
    loss_ic = torch.mean((model(x_ic, t_ic) - config.C0)**2)

    # ---- BC1 (Dirichlet) ----
    t_bc1 = torch.rand(config.n_bc, 1) * config.T
    x_bc1 = torch.zeros(config.n_bc, 1)
    loss_bc1 = torch.mean((model(x_bc1, t_bc1) - config.Cin)**2)

    # ---- BC2 (Neumann) ----
    x_bc2 = torch.ones(config.n_bc, 1) * config.L
    t_bc2 = torch.rand(config.n_bc, 1) * config.T
    x_bc2.requires_grad_(True)
    C_bc2 = model(x_bc2, t_bc2)
    dC_dx_bc2 = torch.autograd.grad(
        C_bc2, x_bc2, torch.ones_like(C_bc2), create_graph=True
    )[0]
    loss_bc2 = torch.mean(dC_dx_bc2**2)

    total = loss_pde + 10 * loss_ic + 10 * loss_bc1 + 10 * loss_bc2
    return total, loss_pde, loss_ic, loss_bc1, loss_bc2

# ============================================================
# TRAIN
# ============================================================
def train(config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINN(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters")

    dt = config.T / config.n_time_steps
    W = load_abc_weights(config.alpha, dt, config.n_time_steps)
    print(f"ABC weights: shape={tuple(W.shape)}, max={W.max():.4e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs
    )

    losses = []
    t0 = time.time()
    for epoch in range(config.n_epochs):
        optimizer.zero_grad()
        total, lp, li, lb1, lb2 = compute_loss(model, config, W)
        total.backward()
        optimizer.step()
        scheduler.step()
        losses.append(total.item())

        if (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch+1:>5}/{config.n_epochs} | "
                  f"total={total.item():.4e} | PDE={lp.item():.4e} | "
                  f"IC={li.item():.4e} | BC1={lb1.item():.4e} | BC2={lb2.item():.4e}")

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")
    return model, losses, elapsed

# ============================================================
# EVALUATE
# ============================================================
def evaluate(model):
    data = np.load("reference_abc.npz")
    x_ref = data['x']
    C_ref = data['C'][:, -1]

    x_t = torch.tensor(x_ref, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)

    with torch.no_grad():
        C_pred = model(x_t, t_t).numpy().flatten()

    mse = np.mean((C_pred - C_ref)**2)
    rmse = np.sqrt(mse)
    mx = np.max(np.abs(C_pred - C_ref))
    return mse, rmse, mx, C_pred, C_ref, x_ref

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PINN with ABC-L1 (Mittag-Leffler kernel)")
    print("=" * 60)

    model, losses, elapsed = train(config)

    torch.save(model.state_dict(), "pinn_abc_l1.pth")
    print("Saved: pinn_abc_l1.pth")

    mse, rmse, mx, C_pred, C_ref, x_ref = evaluate(model)
    print("\n" + "=" * 60)
    print("ERROR METRICS (PINN vs ABC reference)")
    print("=" * 60)
    print(f"  MSE:       {mse:.6e}")
    print(f"  RMSE:      {rmse:.6e}")
    print(f"  Max Error: {mx:.6e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(losses)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss (ABC-L1 PINN)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_ref, C_ref, 'r-', lw=2.5, label='ABC-L1 reference')
    axes[1].plot(x_ref, C_pred, 'b--', lw=2.5, label='ABC-L1 PINN')
    axes[1].set_xlabel("x"); axes[1].set_ylabel("C(x, t=1)")
    axes[1].set_title("ABC-L1 PINN vs reference")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("abc_l1_pinn_result.png", dpi=150)
    plt.show()

    np.savez("abc_l1_pinn_results.npz",
             mse=mse, rmse=rmse, max_err=mx,
             C_pred=C_pred, C_ref=C_ref, x=x_ref,
             elapsed=elapsed)
    print("Saved: abc_l1_pinn_results.npz")