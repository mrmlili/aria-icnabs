"""
ARIA-ICNABS
File: 01_Train_PINN_Caputo_L1_v2.py
Purpose: Caputo-L1 PINN (v2, corrected B(alpha), 4x64, multi-seed).
"""
import torch, torch.nn as nn, numpy as np, time
from scipy.special import gamma

class Config:
    L, T = 1.0, 1.0
    D0, beta, v = 0.01, 0.5, 0.1
    alpha, C0, Cin = 0.8, 0.0, 1.0
    n_hidden, n_neurons = 4, 64
    n_epochs, lr = 8000, 1e-3
    n_ic, n_bc = 300, 300
    n_time_steps = 50
    n_x_colloc = 60

config = Config()

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

def caputo_l1_weights(alpha, dt, Nt):
    B_alpha = 1 - alpha + alpha/gamma(alpha)   # CORRECTED
    scaling = B_alpha / (1 - alpha) / gamma(2 - alpha) / (dt**alpha)
    b = np.array([(k+1)**(1-alpha) - k**(1-alpha) for k in range(Nt+1)])
    return scaling * b

def compute_loss(model, cfg, W):
    Nt = cfg.n_time_steps
    x_pde = torch.rand(cfg.n_x_colloc, 1) * cfg.L
    time_grid = torch.linspace(0, cfg.T, Nt+1)

    history_C = [model(x_pde, time_grid[k].reshape(1,1).expand(cfg.n_x_colloc,1)) for k in range(Nt+1)]

    loss_pde = 0.0
    for n in range(1, Nt+1):
        frac = W[0] * (history_C[n] - history_C[n-1])
        for k in range(1, n):
            frac = frac + W[k] * (history_C[n-k] - history_C[n-k-1])

        x_ad = x_pde.clone().requires_grad_(True)
        t_n = time_grid[n].reshape(1,1).expand(cfg.n_x_colloc,1)
        t_ad = t_n.clone().requires_grad_(True)
        C_ad = model(x_ad, t_ad)

        dC_dx = torch.autograd.grad(C_ad, x_ad, torch.ones_like(C_ad), create_graph=True)[0]
        d2C_dx2 = torch.autograd.grad(dC_dx, x_ad, torch.ones_like(dC_dx), create_graph=True)[0]

        D_x = cfg.D0 * (1 + cfg.beta * x_ad)
        dD_dx = cfg.D0 * cfg.beta
        diffusion = dD_dx * dC_dx + D_x * d2C_dx2
        advection = cfg.v * dC_dx

        residual = frac - diffusion + advection
        loss_pde = loss_pde + torch.mean(residual**2)
    loss_pde = loss_pde / Nt

    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    t_ic = torch.zeros(cfg.n_ic, 1)
    loss_ic = torch.mean((model(x_ic, t_ic) - cfg.C0)**2)

    t_bc1 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_bc1 = torch.zeros(cfg.n_bc, 1)
    loss_bc1 = torch.mean((model(x_bc1, t_bc1) - cfg.Cin)**2)

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
    dt = config.T / config.n_time_steps
    W = torch.tensor(caputo_l1_weights(config.alpha, dt, config.n_time_steps), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)
    t0 = time.time()
    for ep in range(config.n_epochs):
        opt.zero_grad()
        loss = compute_loss(model, config, W)
        loss.backward(); opt.step(); sch.step()
        if (ep+1) % 2000 == 0:
            print(f"    seed={seed} epoch {ep+1}/{config.n_epochs} loss={loss.item():.4e}")
    return model, time.time() - t0

def evaluate(model):
    d = np.load("reference_caputo.npz")
    x_r = d['x']; C_r = d['C'][:, -1]
    x_t = torch.tensor(x_r, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        C_p = model(x_t, t_t).numpy().flatten()
    return np.mean((C_p-C_r)**2), np.sqrt(np.mean((C_p-C_r)**2)), np.max(np.abs(C_p-C_r))

if __name__ == "__main__":
    seeds = [42, 123, 7]
    results = []
    for s in seeds:
        print(f"\n=== Seed {s} ===")
        model, el = train_one_seed(s)
        mse, rmse, mx = evaluate(model)
        print(f"  MSE={mse:.4e}  RMSE={rmse:.4e}  Max={mx:.4e}  Time={el:.1f}s")
        results.append({'seed': s, 'mse': mse, 'rmse': rmse, 'max': mx, 'time': el})
        torch.save(model.state_dict(), f"pinn_caputo_l1_seed{s}.pth")
    np.savez("caputo_l1_results.npz",
             mse=[r['mse'] for r in results],
             rmse=[r['rmse'] for r in results],
             max_err=[r['max'] for r in results],
             times=[r['time'] for r in results])
    print(f"\n=== Summary ===")
    print(f"MSE: {np.mean([r['mse'] for r in results]):.4e} ± {np.std([r['mse'] for r in results]):.4e}")