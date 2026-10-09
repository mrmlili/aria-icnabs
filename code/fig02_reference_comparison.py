"""
F2: Comparison of three reference solutions at t = T = 1.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join('..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load
d_class = np.load('reference_classical.npz')
d_cap   = np.load('reference_caputo.npz')
d_abc   = np.load('reference_abc.npz')

x_class = d_class['x'];  C_class = d_class['C'][:, -1]
x_cap   = d_cap['x'];    C_cap   = d_cap['C'][:, -1]
x_abc   = d_abc['x'];    C_abc   = d_abc['C'][:, -1]

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(x_class, C_class, color='#1f77b4', lw=2.4,
        label='Classical (naive-FD limit)', zorder=3)
ax.plot(x_cap, C_cap, color='#ff7f0e', lw=2.4, ls='--',
        label='Caputo-L1 (power-law kernel)', zorder=2)
ax.plot(x_abc, C_abc, color='#d62728', lw=2.4, ls='-.',
        label='ABC-L1 (Mittag-Leffler kernel)', zorder=1)

ax.set_xlabel('$x$')
ax.set_ylabel('$C(x, t=1)$')
ax.set_title('Reference solutions: three discretizations of the same PDE')
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig02_reference_comparison.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig02_reference_comparison.pdf'))
print(f"✅ Saved: {FIG_DIR}/fig02_reference_comparison.{{png,pdf}}")
