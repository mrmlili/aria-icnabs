"""
F3: Training loss curves for the three residuals (mean over seeds, shaded range).
Requires: retrain_save_losses.py to have been run.
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

SEEDS = [42, 123, 7]

def load_losses(tag, seeds=SEEDS):
    """Load and stack loss arrays across seeds."""
    arrs = []
    for s in seeds:
        fn = f"losses_{tag}_seed{s}.npz"
        if os.path.exists(fn):
            arrs.append(np.load(fn)['losses'])
    return np.array(arrs) if arrs else None

# Naive FD
naive = load_losses('naivefd')
# Caputo
caputo = load_losses('caputo')

# ABC: from existing files
abc_arrs = []
for s in SEEDS:
    if s == 42 and os.path.exists('abc_l1_pinn_results.npz'):
        # seed 42 file doesn't have losses (see earlier), skip
        # If retrained, look for losses_abc_seed42.npz first
        if os.path.exists('losses_abc_seed42.npz'):
            abc_arrs.append(np.load('losses_abc_seed42.npz')['losses'])
    else:
        fn = f"abc_l1_results_seed{s}.npz"
        if os.path.exists(fn):
            d = np.load(fn)
            if 'losses' in d.files:
                abc_arrs.append(d['losses'])
abc = np.array(abc_arrs) if abc_arrs else None

fig, ax = plt.subplots(figsize=(9, 5.5))

def plot_band(ax, arr, color, label):
    if arr is None or len(arr) == 0:
        return
    mean = arr.mean(axis=0)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    epochs = np.arange(1, len(mean) + 1)
    ax.fill_between(epochs, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(epochs, mean, color=color, lw=2.0, label=label)

plot_band(ax, naive,  '#1f77b4', 'Naive FD (local)')
plot_band(ax, caputo, '#ff7f0e', 'Caputo-L1 (power-law)')
plot_band(ax, abc,    '#d62728', 'ABC-L1 (ML kernel)')

ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total loss')
ax.set_title('Training loss (mean over seeds, shaded = min–max range)')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig03_training_loss.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig03_training_loss.pdf'))
print(f"✅ Saved: {FIG_DIR}/fig03_training_loss.{{png,pdf}}")

if naive is None:
    print("  ⚠️  naive FD losses missing — run retrain_save_losses.py first.")
if caputo is None:
    print("  ⚠️  Caputo losses missing — run retrain_save_losses.py first.")
if abc is None or len(abc) < 2:
    print("  ⚠️  ABC losses: only seed 123,7 available.")
