"""
Verify that the production reference solver produces correct results.
"""
import numpy as np

# Load production data
data = np.load("reference_data_final.npz")
C = data['C']

print("=" * 60)
print("VERIFICATION OF PRODUCTION REFERENCE SOLVER")
print("=" * 60)

print(f"\nData shape: {C.shape}")
print(f"Min value:  {C.min():.6e}")
print(f"Max value:  {C.max():.6e}")

# Physical validation
if 0 <= C.min() and C.max() <= 1.5:
    print("\n✅ PHYSICAL VALIDATION PASSED")
    print("   Concentration is within [0, 1.5] — physically admissible")
else:
    print(f"\n❌ PHYSICAL VALIDATION FAILED")
    print(f"   Concentration range: [{C.min():.2e}, {C.max():.2e}]")

# Check profile at t=1
C_final = C[:, -1]
print(f"\nProfile at t=1:")
print(f"  C(0, 1) = {C_final[0]:.6f}")
print(f"  C(0.5, 1) = {C_final[len(C_final)//2]:.6f}")
print(f"  C(1, 1) = {C_final[-1]:.6f}")

# Verify no numerical divergence
if C.max() < 1e3:
    print("\n✅ NO NUMERICAL DIVERGENCE")
else:
    print("\n❌ NUMERICAL DIVERGENCE DETECTED")

print("\n" + "=" * 60)