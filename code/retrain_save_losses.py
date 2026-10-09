"""
ARIA-ICNABS
File: retrain_save_losses.py
Purpose: Re-run Naive FD and Caputo-L1 training with loss history saved.
         ~45 minutes total. Run once.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma

# ============================================================
# SHARED
# ============================================================
class ConfigBase:
    L, T = 1.0, 1.0
    D0, beta, v = 0.01, 0.5, 0.1
    alpha, C0, Cin = 0.8, 0.0, 1.0
    n_hidden, n_neurons = 4, 64
    n_epochs, lr = 8000, 1e-3
    n_ic, n_bc = 300, 300

class ConfigNaive(ConfigBase):
    n_pde, dt_frac = 1000, 0.01

class ConfigCaputo(ConfigBase):
    n_time_steps, n_x_colloc = 50, 60

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

# ============================================================
# NAIVE FD
# ============================================================
def compute_loss_naive(model, cfg):
    B_a = 1 - cfg.alpha + cfg.alpha / gamma(cfg.alpha)
    c_a = B_a / (1 - cfg.alpha)
    x_pde = torch.rand(cfg.n_pde, 1) * cfg.L
    t_pde = torch.rand(cfg.n_pde, 1) * cfg.T
    x_pde.requires_grad_(True); t_pde.requires_grad_(True)
    C = model(x_pde, t_pde)
    dC_dx = torch.autograd.grad(C, x_pde, torch.ones_like(C), create_graph=True)[0]
    d2C_dx2 = torch.autograd.grad(dC_dx, x_pde, torch.ones_like(dC_dx), create_graph=True)[0]
    D_x = cfg.D0 * (1 + cfg.beta * x_pde)
    diff = cfg.D0 * cfg.beta * dC_dx + D_x * d2C_dx2
    adv = cfg.v * dC_dx
    C_t = model(x_pde, t_pde)
    C_tm = model(x_pde, torch.clamp(t_pde - cfg.dt_frac, min=0))
    frac = c_a * (C_t - C_tm) / cfg.dt_frac
    loss_pde = torch.mean((frac - diff + adv)**2)
    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    loss_ic = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - cfg.C0)**2)
    t_bc1 = torch.rand(cfg.n_bc, 1) * cfg.T
    loss_bc1 = torch.mean((model(torch.zeros_like(t_bc1), t_bc1) - cfg.Cin)**2)
    x_bc2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_bc2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_bc2.requires_grad_(True)
    Cb = model(x_bc2, t_bc2)
    dCb = torch.autograd.grad(Cb, x_bc2, torch.ones_like(Cb), create_graph=True)[0]
    loss_bc2 = torch.mean(dCb**2)
    return loss_pde + 10*loss_ic + 10*loss_bc1 + 10*loss_bc2

# ============================================================
# CAPUTO L1
# ============================================================
def caputo_weights(alpha, dt, Nt):
    B_a = 1 - alpha + alpha/gamma(alpha)
    scale = B_a / (1-alpha) / gamma(2-alpha) / (dt**alpha)
    b = np.array([(k+1)**(1-alpha) - k**(1-alpha) for k in range(Nt+1)])
    return scale * b

def compute_loss_caputo(model, cfg, W):
    Nt = cfg.n_time_steps
    x_pde = torch.rand(cfg.n_x_colloc, 1) * cfg.L
    tg = torch.linspace(0, cfg.T, Nt+1)
    hist = [model(x_pde, tg[k].reshape(1,1).expand(cfg.n_x_colloc,1)) for k in range(Nt+1)]
    Lp = 0.0
    for n in range(1, Nt+1):
        frac = W[0] * (hist[n] - hist[n-1])
        for k in range(1, n):
            frac = frac + W[k] * (hist[n-k] - hist[n-k-1])
        x_ad = x_pde.clone().requires_grad_(True)
        t_n = tg[n].reshape(1,1).expand(cfg.n_x_colloc,1)
        t_ad = t_n.clone().requires_grad_(True)
        C = model(x_ad, t_ad)
        dC = torch.autograd.grad(C, x_ad, torch.ones_like(C), create_graph=True)[0]
        d2C = torch.autograd.grad(dC, x_ad, torch.ones_like(dC), create_graph=True)[0]
        D_x = cfg.D0 * (1 + cfg.beta * x_ad)
        diff = cfg.D0 * cfg.beta * dC + D_x * d2C
        adv = cfg.v * dC
        Lp = Lp + torch.mean((frac - diff + adv)**2)
    Lp = Lp / Nt
    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    Li = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - cfg.C0)**2)
    t_b1 = torch.rand(cfg.n_bc, 1) * cfg.T
    Lb1 = torch.mean((model(torch.zeros_like(t_b1), t_b1) - cfg.Cin)**2)
    x_b2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_b2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_b2.requires_grad_(True)
    Cb = model(x_b2, t_b2)
    dCb = torch.autograd.grad(Cb, x_b2, torch.ones_like(Cb), create_graph=True)[0]
    Lb2 = torch.mean(dCb**2)
    return Lp + 10*Li + 10*Lb1 + 10*Lb2

# ============================================================
# TRAIN
# ============================================================
def train_naive(seed):
    cfg = ConfigNaive()
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    losses = []
    for ep in range(cfg.n_epochs):
        opt.zero_grad()
        l = compute_loss_naive(model, cfg)
        l.backward(); opt.step(); sch.step()
        losses.append(l.item())
    return np.array(losses)

def train_caputo(seed):
    cfg = ConfigCaputo()
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN(cfg)
    dt = cfg.T / cfg.n_time_steps
    W = torch.tensor(caputo_weights(cfg.alpha, dt, cfg.n_time_steps), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    losses = []
    for ep in range(cfg.n_epochs):
        opt.zero_grad()
        l = compute_loss_caputo(model, cfg, W)
        l.backward(); opt.step(); sch.step()
        losses.append(l.item())
    return np.array(losses)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    seeds = [42, 123, 7]

    print("=" * 60)
    print("Retraining to save loss histories")
    print("=" * 60)

    print("\n[1/2] Naive FD (~3 min total)")
    for s in seeds:
        t0 = time.time()
        print(f"  seed={s}...", end=" ", flush=True)
        losses = train_naive(s)
        np.savez(f"losses_naivefd_seed{s}.npz", losses=losses)
        print(f"done in {time.time()-t0:.0f}s, final={losses[-1]:.4e}")

    print("\n[2/2] Caputo-L1 (~40 min total)")
    for s in seeds:
        t0 = time.time()
        print(f"  seed={s}...", end=" ", flush=True)
        losses = train_caputo(s)
        np.savez(f"losses_caputo_seed{s}.npz", losses=losses)
        print(f"done in {time.time()-t0:.0f}s, final={losses[-1]:.4e}")

    print("\n✅ Loss histories saved.")
    print("   losses_naivefd_seed{42,123,7}.npz")
    print("   losses_caputo_seed{42,123,7}.npz")