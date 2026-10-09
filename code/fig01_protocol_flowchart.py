"""
F1: Diagnostic Protocol Flowchart
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join('..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis('off')

def box(x, y, w, h, text, fc='#E8F0FE', ec='#1A73E8', fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)

def arrow(x1, y1, x2, y2, color='black'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=15, color=color, linewidth=1.6))

# Input
box(3, 12.7, 4, 0.8, 'PINN residual $\\mathcal{R}[C]$', fc='#FFF4E5', ec='#E67E22')
arrow(5, 12.7, 5, 12.1)

# Test 1
box(2.2, 11.1, 5.6, 0.9, 'Test 1: Kernel presence?\n(convolution or history sum)')
arrow(5, 11.1, 5, 10.5)
arrow(7.8, 11.55, 8.8, 11.55, color='#C0392B')
ax.text(7.1, 11.72, 'fail', fontsize=10, color='#C0392B')
box(7.0, 10.9, 2.8, 1.3, 'LOCAL\nSURROGATE', fc='#FDECEA', ec='#C0392B', fs=10)
ax.text(4.3, 10.3, 'pass', fontsize=10, color='#27AE60')

# Test 2
box(2.2, 8.9, 5.6, 0.9, 'Test 2: $\\Delta t$-refinement\nconverges to classical?')
arrow(5, 8.9, 5, 8.3)
arrow(7.8, 9.35, 8.8, 9.35, color='#C0392B')
ax.text(7.1, 9.52, 'yes', fontsize=10, color='#C0392B')
box(7.0, 8.7, 2.8, 1.3, 'LOCAL\nSURROGATE', fc='#FDECEA', ec='#C0392B', fs=10)
ax.text(4.3, 8.1, 'no', fontsize=10, color='#27AE60')

# Test 3
box(2.2, 6.7, 5.6, 0.9, 'Test 3: matches\n$C_1(x,\\, T/c(\\alpha))$?')
arrow(5, 6.7, 5, 6.1)
arrow(7.8, 7.15, 8.8, 7.15, color='#C0392B')
ax.text(7.1, 7.32, 'yes', fontsize=10, color='#C0392B')
box(7.0, 6.5, 2.8, 1.3, 'LOCAL\nSURROGATE', fc='#FDECEA', ec='#C0392B', fs=10)
ax.text(4.3, 5.9, 'no', fontsize=10, color='#27AE60')

# Final
arrow(5, 6.1, 5, 5.5)
box(2.2, 4.2, 5.6, 1.2, 'GENUINE NONLOCAL\n(power-law or Mittag-Leffler kernel)',
    fc='#E8F5E9', ec='#27AE60', fs=11)

ax.text(5, 3.2, 'Classification outcome:', ha='center', fontsize=11, style='italic')
ax.text(5, 2.6, '$\\bullet$ Local surrogate    '
             '$\\bullet$ Power-law nonlocal    '
             '$\\bullet$ ML-kernel nonlocal',
        ha='center', fontsize=10, color='#333')

plt.savefig(os.path.join(FIG_DIR, 'fig01_protocol_flowchart.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig01_protocol_flowchart.pdf'))
print(f"✅ Saved: {FIG_DIR}/fig01_protocol_flowchart.{{png,pdf}}")
