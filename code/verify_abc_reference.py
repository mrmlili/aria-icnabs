"""
ARIA-ICNABS Project
File: verify_abc_reference.py
Purpose: Verify that ABC reference solution is physically valid
         and compare with Caputo reference.
"""
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("VERIFICATION OF ABC REFERENCE SOLUTION")
print("=" * 70)

# --- Load ABC reference ---
data_abc = np.load("reference_abc.npz")
x = data_abc['x']
C_abc = data_abc['C']
alpha = float(data_abc['alpha'])
Nt = int(data_abc['Nt'])

print(f"\nABC reference: alpha = {alpha}, Nt = {Nt}")
print(f"  Shape: {C_abc.shape}")
print(f"  Min:  {C_abc.min():.6e}")
print(f"  Max:  {C_abc.max():.6e}")

# --- Physical validation ---
print("\n" + "=" * 70)
print("PHYSICAL VALIDATION")
print("=" * 70)

if C_abc.min() >= -1e-6 and C_abc.max() <= 1.5:
    print("✅ PASSED: concentration in [0, 1.5] — physically admissible")
else:
    print(f"❌ FAILED: concentration range [{C_abc.min():.2e}, {C_abc.max():.2e}]")
    print("   → the ABC solver may be unstable; check weights or time-stepping")

# Check monotonicity in x (should be decreasing from C_in=1 to some value)
C_final = C_abc[:, -1]
monotonic_ok = np.all(np.diff(C_final) <= 1e-3)  # allow small numerical wiggle
if monotonic_ok:
    print("✅ PASSED: profile C(x,T) is monotonically decreasing in x")
else:
    print("⚠️  WARNING: profile has non-monotonic features — inspect")

# Check boundary conditions
print(f"\nBoundary values at t=T:")
print(f"  C(0, T) = {C_abc[0, -1]:.6f}  (should be ≈ Cin = 1.0)")
print(f"  C(L, T) = {C_abc[-1, -1]:.6f}  (should be in [0, 1))")
print(f"  C(L/2, T) = {C_abc[len(x)//2, -1]:.6f}")

# --- Compare with Caputo reference if available ---
import os
if os.path.exists("reference_data_final.npz"):
    print("\n" + "=" * 70)
    print("COMPARISON: ABC vs Caputo L1 reference")
    print("=" * 70)
    
    data_cap = np.load("reference_data_final.npz")
    x_cap = data_cap['x']
    C_cap = data_cap['C'][:, -1]  # Caputo at t=T
    
    # Interpolate ABC solution onto Caputo x grid (or vice versa)
    C_abc_interp = np.interp(x_cap, x, C_abc[:, -1])
    
    diff = C_abc_interp - C_cap
    mse = np.mean(diff**2)
    max_diff = np.max(np.abs(diff))
    
    print(f"  Caputo shape: {C_cap.shape}, ABC shape (interp): {C_abc_interp.shape}")
    print(f"  MSE(ABC, Caputo) = {mse:.6e}")
    print(f"  Max |diff|        = {max_diff:.6e}")
    print(f"  Caputo max        = {C_cap.max():.6f}")
    print(f"  ABC max           = {C_abc[:, -1].max():.6f}")
    
    if mse > 1e-4:
        print("  ✅ ABC and Caputo references differ significantly")
        print("     (as expected — kernel types are fundamentally different)")
    else:
        print("  ⚠️  ABC and Caputo references are very close")
        print("     (this would be suspicious for such different kernels)")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_cap, C_cap, 'b-', linewidth=2.5, label='Caputo L1 (power-law kernel)')
    ax.plot(x, C_abc[:, -1], 'r--', linewidth=2.5, label='ABC L1 (Mittag-Leffler kernel)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('C(x, t=1.0)', fontsize=12)
    ax.set_title('Reference comparison: Caputo vs ABC (alpha=0.8)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reference_comparison.png', dpi=150)
    plt.show()
    print("\n  → Saved: reference_comparison.png")
else:
    print("\n⚠️  No Caputo reference (reference_data_final.npz) found.")
    print("   Skipping comparison.")

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If all checks above passed, the ABC reference is trustworthy")
print("and we can proceed to training a PINN against it.")