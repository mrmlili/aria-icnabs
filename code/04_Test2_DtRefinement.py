"""
ARIA-ICNABS — Test 2: Δt-refinement study.
Trains naive FD PINN with progressively smaller Δt and
verifies convergence of the solution to the classical limit.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import gamma

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join('..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
class Config:
    L, T = 1.0, 1.0
    D0, beta, v = 0.01, 0.5, 0.1
    alpha, C0, Cin = 0.8, 0.0, 1.0
    n_hidden, n_neurons = 4, 64
    n_epochs, lr = 6000, 1e-3
    n_pde, n_ic, n_bc = 1000, 300, 300

config = Config()
B_alpha = 1 - config.alpha + config.alpha / gamma(config.alpha)
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

def compute_loss(model, cfg, dt_frac):
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
    C_tm = model(x_pde, torch.clamp(t_pde - dt_frac, min=0))
    frac = c_alpha * (C_t - C_tm) / dt_frac
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

def train(dt_frac, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)
    t0 = time.time()
    for ep in range(config.n_epochs):
        opt.zero_grad()
        l = compute_loss(model, config, dt_frac)
        l.backward(); opt.step(); sch.step()
    return model, time.time() - t0

def evaluate(model):
    d = np.load("reference_classical.npz")
    x_r = d['x']; C_r = d['C'][:, -1]
    x_t = torch.tensor(x_r, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        C_p = model(x_t, t_t).numpy().flatten()
    return float(np.mean((C_p - C_r)**2)), float(np.max(np.abs(C_p - C_r)))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    dt_values = [0.1, 0.05, 0.025, 0.0125]
    results = []
    print("=" * 60)
    print("TEST 2: Δt-refinement study")
    print("=" * 60)
    for dt in dt_values:
        print(f"\n--- Δt = {dt} ---")
        model, elapsed = train(dt)
        mse, maxerr = evaluate(model)
        print(f"  MSE = {mse:.4e}  |  MaxErr = {maxerr:.4e}  |  time = {elapsed:.0f}s")
        results.append({'dt': dt, 'mse': mse, 'max': maxerr, 'time': elapsed})
        torch.save(model.state_dict(), f"pinn_naivefd_dt{dt}.pth")

    np.savez("test2_dt_refinement.npz",
             dt=[r['dt'] for r in results],
             mse=[r['mse'] for r in results],
             max_err=[r['max'] for r in results],
             times=[r['time'] for r in results])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dts = np.array([r['dt'] for r in results])
    mses = np.array([r['mse'] for r in results])
    maxes = np.array([r['max'] for r in results])

    axes[0].loglog(dts, mses, 'o-', color='#1f77b4', lw=2, ms=8)
    axes[0].set_xlabel(r'$\Delta t$')
    axes[0].set_ylabel('MSE vs classical reference')
    axes[0].set_title('Test 2: MSE convergence')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].invert_xaxis()

    axes[1].loglog(dts, maxes, 's-', color='#d62728', lw=2, ms=8)
    axes[1].set_xlabel(r'$\Delta t$')
    axes[1].set_ylabel('Max error vs classical reference')
    axes[1].set_title('Test 2: Max-error convergence')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig07_dt_refinement.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig07_dt_refinement.pdf'))
    print(f"\n✅ Saved: {FIG_DIR}/fig07_dt_refinement.{{png,pdf}}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  Δt = {r['dt']:.4f}  →  MSE = {r['mse']:.4e},  MaxErr = {r['max']:.4e}")