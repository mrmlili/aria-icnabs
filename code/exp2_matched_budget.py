"""
Experiment 2: Matched-budget comparison for Naive FD.
Original: n_pde = 1000
This run: n_pde = 3000 (matched to Caputo/ABC: 50x60 = 3000)
"""
import torch
import torch.nn as nn
import numpy as np
import time
from scipy.special import gamma


class Config:
    L, T = 1.0, 1.0
    D0, beta, v = 0.01, 0.5, 0.1
    alpha, C0, Cin = 0.8, 0.0, 1.0
    n_hidden, n_neurons = 4, 64
    n_epochs, lr = 8000, 1e-3
    n_pde = 3000          # <-- the only change
    n_ic, n_bc = 300, 300
    dt_frac = 0.01

config = Config()
B_alpha = 1 - config.alpha + config.alpha / gamma(config.alpha)
c_alpha = B_alpha / (1 - config.alpha)
print(f"c(alpha) = {c_alpha:.6f}")


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
    x_pde = torch.rand(cfg.n_pde, 1) * cfg.L
    t_pde = torch.rand(cfg.n_pde, 1) * cfg.T
    x_pde.requires_grad_(True)
    t_pde.requires_grad_(True)

    C = model(x_pde, t_pde)
    dC_dx = torch.autograd.grad(C, x_pde, torch.ones_like(C), create_graph=True)[0]
    d2C_dx2 = torch.autograd.grad(dC_dx, x_pde, torch.ones_like(dC_dx), create_graph=True)[0]

    D_x = cfg.D0 * (1 + cfg.beta * x_pde)
    diff = cfg.D0 * cfg.beta * dC_dx + D_x * d2C_dx2
    adv = cfg.v * dC_dx

    C_t = model(x_pde, t_pde)
    C_tm = model(x_pde, torch.clamp(t_pde - cfg.dt_frac, min=0))
    frac = c_alpha * (C_t - C_tm) / cfg.dt_frac

    loss_pde = torch.mean((frac - diff + adv) ** 2)

    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    loss_ic = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - cfg.C0) ** 2)

    t_bc1 = torch.rand(cfg.n_bc, 1) * cfg.T
    loss_bc1 = torch.mean((model(torch.zeros_like(t_bc1), t_bc1) - cfg.Cin) ** 2)

    x_bc2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_bc2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_bc2.requires_grad_(True)
    Cb = model(x_bc2, t_bc2)
    dCb = torch.autograd.grad(Cb, x_bc2, torch.ones_like(Cb), create_graph=True)[0]
    loss_bc2 = torch.mean(dCb ** 2)

    return loss_pde + 10 * loss_ic + 10 * loss_bc1 + 10 * loss_bc2


def train_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PINN(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)
    t0 = time.time()
    for ep in range(config.n_epochs):
        opt.zero_grad()
        l = compute_loss(model, config)
        l.backward()
        opt.step()
        sch.step()
    return model, time.time() - t0


def evaluate(model):
    d = np.load("reference_classical.npz")
    x_r = d['x']
    C_r = d['C'][:, -1]
    x_t = torch.tensor(x_r, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        C_p = model(x_t, t_t).numpy().flatten()
    mse = float(np.mean((C_p - C_r) ** 2))
    rmse = float(np.sqrt(mse))
    mx = float(np.max(np.abs(C_p - C_r)))
    return mse, rmse, mx


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2: Matched-budget Naive FD")
    print(f"  n_pde = {config.n_pde}  (original was 1000)")
    print("=" * 60)

    seeds = [42, 123, 7]
    results = []
    for s in seeds:
        print(f"\n--- seed = {s} ---")
        model, el = train_one_seed(s)
        mse, rmse, mx = evaluate(model)
        print(f"  MSE = {mse:.4e}  RMSE = {rmse:.4e}  Max = {mx:.4e}  Time = {el:.0f}s")
        results.append({'seed': s, 'mse': mse, 'rmse': rmse, 'mx': mx, 'time': el})
        torch.save(model.state_dict(), f"pinn_naivefd_matched_seed{s}.pth")

    mses = np.array([r['mse'] for r in results])
    times = np.array([r['time'] for r in results])
    print("\n" + "=" * 60)
    print("SUMMARY: Matched-budget Naive FD")
    print("=" * 60)
    print(f"  MSE (mean +/- std) = {mses.mean():.4e} +/- {mses.std():.4e}")
    print(f"  Time (mean)        = {times.mean():.0f}s")
    print(f"\n  Original (n_pde=1000): MSE = 3.04e-6 +/- 1.42e-6")
    print(f"  Ratio matched/orig:    {mses.mean()/3.04e-6:.2f}x")

    np.savez("exp2_matched_budget.npz",
             seeds=[r['seed'] for r in results],
             mse=mses,
             rmse=[r['rmse'] for r in results],
             max_err=[r['mx'] for r in results],
             times=times)
    print("\nSaved: exp2_matched_budget.npz")