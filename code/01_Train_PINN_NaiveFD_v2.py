"""
ARIA-ICNABS
File: 01_Train_PINN_NaiveFD_v2.py
Purpose: Naive FD PINN (v2, 4x64, multi-seed).
"""
import torch, torch.nn as nn, numpy as np, time
from scipy.special import gamma

class Config:
    L, T = 1.0, 1.0
    D0, beta, v = 0.01, 0.5, 0.1
    alpha, C0, Cin = 0.8, 0.0, 1.0
    n_hidden, n_neurons = 4, 64
    n_epochs, lr = 8000, 1e-3
    n_pde, n_ic, n_bc = 1000, 300, 300
    dt_frac = 0.01

config = Config()
B_alpha = 1 - config.alpha + config.alpha/gamma(config.alpha)
c_alpha = B_alpha / (1 - config.alpha)

class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = [nn.Linear(2, cfg.n_neurons), nn.Tanh()]
        for _ in range(cfg.n_hidden - 1):
            layers += [nn.Linear(cfg.n_neurons, cfg.n_neurons), nn.Tanh()]
        layers.append(nn.Linear(cfg.n_neurons, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def compute_loss(model, cfg):
    # PDE
    x_pde = torch.rand(cfg.n_pde, 1) * cfg.L
    t_pde = torch.rand(cfg.n_pde, 1) * cfg.T
    x_pde.requires_grad_(True); t_pde.requires_grad_(True)

    C = model(x_pde, t_pde)
    dC_dx = torch.autograd.grad(C, x_pde, torch.ones_like(C), create_graph=True)[0]
    d2C_dx2 = torch.autograd.grad(dC_dx, x_pde, torch.ones_like(dC_dx), create_graph=True)[0]

    D_x = cfg.D0 * (1 + cfg.beta * x_pde)
    dD_dx = cfg.D0 * cfg.beta
    diffusion = dD_dx * dC_dx + D_x * d2C_dx2
    advection = cfg.v * dC_dx

    # Naive FD
    C_t = model(x_pde, t_pde)
    C_t_minus = model(x_pde, torch.clamp(t_pde - cfg.dt_frac, min=0))
    dC_dt = (C_t - C_t_minus) / cfg.dt_frac
    frac = c_alpha * dC_dt

    residual = frac - diffusion + advection
    loss_pde = torch.mean(residual**2)

    # IC
    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    t_ic = torch.zeros(cfg.n_ic, 1)
    loss_ic = torch.mean((model(x_ic, t_ic) - cfg.C0)**2)

    # BC1 (Dirichlet)
    t_bc1 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_bc1 = torch.zeros(cfg.n_bc, 1)
    loss_bc1 = torch.mean((model(x_bc1, t_bc1) - cfg.Cin)**2)

    # BC2 (Neumann)
    x_bc2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_bc2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_bc2.requires_grad_(True)
    C_bc2 = model(x_bc2, t_bc2)
    dC_dx_bc2 = torch.autograd.grad(C_bc2, x_bc2, torch.ones_like(C_bc2), create_graph=True)[0]
    loss_bc2 = torch.mean(dC_dx_bc2**2)

    total = loss_pde + 10*loss_ic + 10*loss_bc1 + 10*loss_bc2
    return total

def train_one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)
    t0 = time.time()
    losses = []
    for ep in range(config.n_epochs):
        opt.zero_grad()
        loss = compute_loss(model, config)
        loss.backward(); opt.step(); sch.step()
        losses.append(loss.item())
        if (ep+1) % 2000 == 0:
            print(f"    seed={seed} epoch {ep+1}/{config.n_epochs} loss={loss.item():.4e}")
    return model, losses, time.time() - t0

def evaluate(model):
    d = np.load("reference_classical.npz")
    x_r = d['x']; C_r = d['C'][:, -1]
    x_t = torch.tensor(x_r, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        C_p = model(x_t, t_t).numpy().flatten()
    return np.mean((C_p - C_r)**2), np.sqrt(np.mean((C_p - C_r)**2)), np.max(np.abs(C_p - C_r))

if __name__ == "__main__":
    seeds = [42, 123, 7]
    results = []
    for s in seeds:
        print(f"\n=== Seed {s} ===")
        model, losses, el = train_one_seed(s)
        mse, rmse, mx = evaluate(model)
        print(f"  MSE={mse:.4e}  RMSE={rmse:.4e}  Max={mx:.4e}  Time={el:.1f}s")
        results.append({'seed': s, 'mse': mse, 'rmse': rmse, 'max': mx, 'time': el})
        torch.save(model.state_dict(), f"pinn_naivefd_seed{s}.pth")
    np.savez("naivefd_results.npz",
             mse=[r['mse'] for r in results],
             rmse=[r['rmse'] for r in results],
             max_err=[r['max'] for r in results],
             times=[r['time'] for r in results])
    print(f"\n=== Summary ===")
    print(f"MSE: {np.mean([r['mse'] for r in results]):.4e} ± {np.std([r['mse'] for r in results]):.4e}")