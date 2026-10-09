"""
ARIA-ICNABS — Test 3: Effective-time verification.
For each α ∈ {0.5, 0.8, 0.9}, train naive FD PINN and compare
its prediction at t=1 with C_1(x, 1/c(α)).
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.interpolate import interp1d

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join('..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

class PINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def train_pinn(alpha, seed=42, n_epochs=6000):
    torch.manual_seed(seed); np.random.seed(seed)
    B_a = 1 - alpha + alpha/gamma(alpha)
    c_a = B_a / (1 - alpha)
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        opt.zero_grad()
        x_pde = torch.rand(1000, 1); t_pde = torch.rand(1000, 1)
        x_pde.requires_grad_(True); t_pde.requires_grad_(True)
        C = model(x_pde, t_pde)
        dC_dx = torch.autograd.grad(C, x_pde, torch.ones_like(C), create_graph=True)[0]
        d2C_dx2 = torch.autograd.grad(dC_dx, x_pde, torch.ones_like(dC_dx), create_graph=True)[0]
        D_x = 0.01 * (1 + 0.5 * x_pde)
        diff = 0.01*0.5*dC_dx + D_x * d2C_dx2
        adv = 0.1 * dC_dx
        C_t = model(x_pde, t_pde)
        C_tm = model(x_pde, torch.clamp(t_pde - 0.01, min=0))
        frac = c_a * (C_t - C_tm) / 0.01
        Lp = torch.mean((frac - diff + adv)**2)
        x_ic = torch.rand(300, 1)
        Li = torch.mean((model(x_ic, torch.zeros_like(x_ic)))**2)
        t_bc1 = torch.rand(300, 1)
        Lb1 = torch.mean((model(torch.zeros_like(t_bc1), t_bc1) - 1.0)**2)
        x_bc2 = torch.ones(300, 1)
        t_bc2 = torch.rand(300, 1)
        x_bc2.requires_grad_(True)
        Cb = model(x_bc2, t_bc2)
        dCb = torch.autograd.grad(Cb, x_bc2, torch.ones_like(Cb), create_graph=True)[0]
        Lb2 = torch.mean(dCb**2)
        loss = Lp + 10*Li + 10*Lb1 + 10*Lb2
        loss.backward(); opt.step(); sch.step()
    return model, c_a

def eval_pinn_at_t1(model, x_grid):
    model.eval()
    x_t = torch.tensor(x_grid, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        return model(x_t, t_t).numpy().flatten()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Load base reference C_1
    base = np.load("reference_base_c1.npz")
    x_base = base['x']
    C1_base = base['C']  # shape (Nx+1, Nt+1)
    t_grid = base['t']

    def C1_at(tau):
        """Interpolate C_1(x, tau) at arbitrary tau."""
        # Find nearest index
        idx = np.argmin(np.abs(t_grid - tau))
        return C1_base[:, idx]

    alphas = [0.5, 0.8, 0.9]
    results = []
    print("=" * 60)
    print("TEST 3: Effective-time verification")
    print("=" * 60)

    for alpha in alphas:
        B_a = 1 - alpha + alpha/gamma(alpha)
        c_a = B_a / (1 - alpha)
        tau = 1.0 / c_a
        print(f"\n--- α = {alpha}, c(α) = {c_a:.4f}, τ_eff = {tau:.4f} ---")

        model, _ = train_pinn(alpha)
        C_pinn = eval_pinn_at_t1(model, x_base)
        C_base = C1_at(tau)

        mse = float(np.mean((C_pinn - C_base)**2))
        maxerr = float(np.max(np.abs(C_pinn - C_base)))
        print(f"  MSE(PINN vs C_1(·, {tau:.4f})) = {mse:.4e}")
        print(f"  MaxErr = {maxerr:.4e}")
        results.append({'alpha': alpha, 'c': c_a, 'tau': tau,
                        'mse': mse, 'max': maxerr,
                        'C_pinn': C_pinn, 'C_base': C_base})

    np.savez("test3_effective_time.npz",
             alphas=[r['alpha'] for r in results],
             taus=[r['tau'] for r in results],
             mse=[r['mse'] for r in results],
             max_err=[r['max'] for r in results])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, r in zip(axes, results):
        ax.plot(x_base, r['C_base'], color='#2ca02c', lw=2.4,
                label=fr'$C_1(x, \tau={r["tau"]:.3f})$')
        ax.plot(x_base, r['C_pinn'], 'k--', lw=2.0,
                label=fr'PINN ($\alpha={r["alpha"]}$)')
        ax.set_xlabel('$x$')
        ax.set_title(fr'$\alpha={r["alpha"]}$, $\tau=1/c(\alpha)={r["tau"]:.3f}$' + 
                     f'\nMSE = {r["mse"]:.2e}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel('$C(x, t=1)$')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig08_effective_time.png'))
    plt.savefig(os.path.join(FIG_DIR, 'fig08_effective_time.pdf'))
    print(f"\n✅ Saved: {FIG_DIR}/fig08_effective_time.{{png,pdf}}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'α':>5} | {'c(α)':>8} | {'τ_eff':>8} | {'MSE':>12} | {'MaxErr':>12}")
    for r in results:
        print(f"{r['alpha']:>5.1f} | {r['c']:>8.3f} | {r['tau']:>8.3f} | "
              f"{r['mse']:>12.4e} | {r['max']:>12.4e}")