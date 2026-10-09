"""
ARIA-ICNABS — Figure 5 (CORRECTED)
Bug fix: previously compared C_cap with C_cap (always zero).
         Now compares PINN prediction against the correct reference.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
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

def eval_pinn(ckpt, x_grid):
    model = PINN(4, 64)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    x_t = torch.tensor(x_grid, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        return model(x_t, t_t).numpy().flatten()

# --- References ---
d_class = np.load('reference_classical.npz')
d_cap   = np.load('reference_caputo.npz')
d_abc   = np.load('reference_abc.npz')

x_class = d_class['x'];  C_class_ref = d_class['C'][:, -1]
x_cap   = d_cap['x'];    C_cap_ref   = d_cap['C'][:, -1]
x_abc   = d_abc['x'];    C_abc_ref   = d_abc['C'][:, -1]

# --- PINN predictions ---
C_naive_pinn = eval_pinn('pinn_naivefd_seed42.pth', x_class)
C_cap_pinn   = eval_pinn('pinn_caputo_l1_seed42.pth', x_cap)
C_abc_pinn   = eval_pinn('pinn_abc_l1.pth', x_abc)

# --- Correct error profiles ---
err_naive = np.abs(C_naive_pinn - C_class_ref)
err_cap   = np.abs(C_cap_pinn   - C_cap_ref)
err_abc   = np.abs(C_abc_pinn   - C_abc_ref)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

panels = [
    (axes[0], x_class, err_naive, 'Naive FD', '#1f77b4'),
    (axes[1], x_cap,   err_cap,   'Caputo-L1', '#ff7f0e'),
    (axes[2], x_abc,   err_abc,   'ABC-L1', '#d62728'),
]

for ax, xg, err, name, col in panels:
    ax.plot(xg, err, color=col, lw=2.0)
    ax.fill_between(xg, 0, err, color=col, alpha=0.15)
    ax.set_xlabel('$x$')
    ax.set_title(f'{name}  —  max = {err.max():.2e}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

axes[0].set_ylabel(r'$|C_{\mathrm{PINN}}(x,1) - C_{\mathrm{ref}}(x,1)|$')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig05_error_profiles.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig05_error_profiles.pdf'))
print(f"✅ Saved: {FIG_DIR}/fig05_error_profiles.{{png,pdf}}")
print(f"\nMax errors (seed 42):")
print(f"  Naive FD:  {err_naive.max():.4e}")
print(f"  Caputo-L1: {err_cap.max():.4e}")
print(f"  ABC-L1:    {err_abc.max():.4e}")