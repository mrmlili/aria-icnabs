"""
F6: MSE and training-time bar charts for the three residuals.
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

# Load aggregates
nfd = np.load('naivefd_results.npz')
cap = np.load('caputo_l1_results.npz')
abc = np.load('abc_l1_multi_seed_results.npz')

names = ['Naive FD\n(local)', 'Caputo-L1\n(power-law)', 'ABC-L1\n(ML kernel)']
mse_mean = np.array([nfd['mse'].mean(),   cap['mse'].mean(),   abc['mse'].mean()])
mse_std  = np.array([nfd['mse'].std(),    cap['mse'].std(),    abc['mse'].std()])
time_mean = np.array([nfd['times'].mean(), cap['times'].mean(), abc['times'].mean()])
time_std  = np.array([nfd['times'].std(),  cap['times'].std(),  abc['times'].std()])

colors = ['#1f77b4', '#ff7f0e', '#d62728']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel 1: MSE ---
ax = axes[0]
bars = ax.bar(names, mse_mean, yerr=mse_std, color=colors,
              edgecolor='black', linewidth=0.8, capsize=4)
ax.set_yscale('log')
ax.set_ylabel('MSE (mean ± std over 3 seeds)')
ax.set_title('Accuracy')
ax.grid(True, alpha=0.3, axis='y', which='both')

# Annotate ratios
baseline = mse_mean[0]
for i, (b, m) in enumerate(zip(bars, mse_mean)):
    ratio = m / baseline
    ax.text(b.get_x() + b.get_width()/2, m * 1.4,
            f'{ratio:.1f}×', ha='center', fontsize=10)

# --- Panel 2: Training time ---
ax = axes[1]
bars = ax.bar(names, time_mean, yerr=time_std, color=colors,
              edgecolor='black', linewidth=0.8, capsize=4)
ax.set_ylabel('Training time (s, mean ± std)')
ax.set_title('Computational cost')
ax.grid(True, alpha=0.3, axis='y')

for i, (b, t) in enumerate(zip(bars, time_mean)):
    # قرار دادن برچسب بالاتر از error bar cap
    y_pos = t + time_std[i] + 80
    ax.text(b.get_x() + b.get_width()/2, y_pos,
            f'{t:.0f}s', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig06_mse_cost_bars.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig06_mse_cost_bars.pdf'))
print(f"✅ Saved: {FIG_DIR}/fig06_mse_cost_bars.{{png,pdf}}")


