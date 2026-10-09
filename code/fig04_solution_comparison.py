"""
F4: Three-panel comparison: PINN prediction vs reference for each residual at t=1.
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

# References
d_class = np.load('reference_classical.npz')
d_cap   = np.load('reference_caputo.npz')
d_abc   = np.load('reference_abc.npz')

x_class = d_class['x'];  C_class = d_class['C'][:, -1]
x_cap   = d_cap['x'];    C_cap   = d_cap['C'][:, -1]
x_abc   = d_abc['x'];    C_abc   = d_abc['C'][:, -1]

# PINN predictions (seed 42 for all three)
C_naive_pinn = eval_pinn('pinn_naivefd_seed42.pth', x_class)
C_cap_pinn   = eval_pinn('pinn_caputo_l1_seed42.pth', x_cap)
C_abc_pinn   = eval_pinn('pinn_abc_l1.pth', x_abc)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

panels = [
    (axes[0], x_class, C_class, C_naive_pinn,
     'Naive FD', '#1f77b4'),
    (axes[1], x_cap, C_cap, C_cap_pinn,
     'Caputo-L1', '#ff7f0e'),
    (axes[2], x_abc, C_abc, C_abc_pinn,
     'ABC-L1', '#d62728'),
]

for ax, xg, Cref, Cpinn, name, col in panels:
    ax.plot(xg, Cref, color=col, lw=2.4, label=f'{name} reference')
    ax.plot(xg, Cpinn, color='black', lw=2.0, ls='--', label=f'{name} PINN')
    ax.set_xlabel('$x$')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)

axes[0].set_ylabel('$C(x, t=1)$')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig04_solution_comparison.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig04_solution_comparison.pdf'))
print(f"✅ Saved: {FIG_DIR}/fig04_solution_comparison.{{png,pdf}}")
