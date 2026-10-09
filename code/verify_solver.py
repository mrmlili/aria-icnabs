"""
ARIA-ICNABS - Definitive Verification of L1 Reference Solver
Compares correct vs buggy formulations on a simple test problem.
"""
import numpy as np
from scipy.special import gamma
from scipy.linalg import solve_banded

# ============================================================
# TEST PROBLEM PARAMETERS
# ============================================================
L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
alpha = 0.8
C0, Cin = 0.0, 1.0

Nx, Nt = 100, 500  # smaller for verification
dx, dt = L/Nx, T/Nt
x = np.linspace(0, L, Nx+1)

def l1_weights(alpha, dt, Nt):
    B_alpha = 1 - alpha + alpha/gamma(alpha )
    scaling = B_alpha / (1 - alpha) / gamma(2 - alpha) / (dt**alpha)
    b = np.array([(k+1)**(1-alpha) - k**(1-alpha) for k in range(Nt+1)])
    return scaling * b

def solve_correct():
    """CORRECT L1 formulation (used in production)."""
    C = np.zeros((Nx+1, Nt+1))
    C[:, 0] = C0
    C[0, :] = Cin
    weights = l1_weights(alpha, dt, Nt)
    D = D0 * (1 + beta * x)
    
    a = np.zeros(Nx-1); b = np.zeros(Nx-1); c = np.zeros(Nx-1)
    for i in range(1, Nx):
        a[i-1] = -(D[i]/dx**2 + v/(2*dx))
        b[i-1] = weights[0] + 2*D[i]/dx**2
        c[i-1] = -(D[i]/dx**2 - v/(2*dx))
    b[-1] = b[-1] + c[-1]
    ab = np.zeros((3, Nx-1))
    ab[0, 1:] = c[:-1]; ab[1, :] = b; ab[2, :-1] = a[1:]
    
    for n in range(1, Nt+1):
        d = np.zeros(Nx-1)
        for i in range(1, Nx):
            rhs = 0.0
            # CORRECT formulation:
            for k in range(1, n):
                rhs += (weights[k-1] - weights[k]) * C[i, n-k]
            rhs += weights[n-1] * C[i, 0]
            d[i-1] = rhs
        d[0] += -a[0] * C[0, n]
        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx-1, n]
    return C

def solve_buggy():
    """BUGGY formulation (from deprecated prototype)."""
    C = np.zeros((Nx+1, Nt+1))
    C[:, 0] = C0
    C[0, :] = Cin
    weights = l1_weights(alpha, dt, Nt)
    D = D0 * (1 + beta * x)
    
    a = np.zeros(Nx-1); b = np.zeros(Nx-1); c = np.zeros(Nx-1)
    for i in range(1, Nx):
        a[i-1] = -(D[i]/dx**2 + v/(2*dx))
        b[i-1] = weights[0] + 2*D[i]/dx**2
        c[i-1] = -(D[i]/dx**2 - v/(2*dx))
    b[-1] = b[-1] + c[-1]
    ab = np.zeros((3, Nx-1))
    ab[0, 1:] = c[:-1]; ab[1, :] = b; ab[2, :-1] = a[1:]
    
    for n in range(1, Nt+1):
        d = np.zeros(Nx-1)
        for i in range(1, Nx):
            rhs = 0.0
            # BUGGY formulation:
            for k in range(1, n+1):
                if n-k >= 0:
                    rhs += weights[k] * C[i, n-k]
            rhs -= weights[n] * C[i, 0]
            d[i-1] = rhs
        d[0] += -a[0] * C[0, n]
        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx-1, n]
    return C

# ============================================================
# RUN AND COMPARE
# ============================================================
print("=" * 70)
print("VERIFICATION: CORRECT vs BUGGY L1 FORMULATION")
print("=" * 70)

C_correct = solve_correct()
C_buggy = solve_buggy()

print("\n✅ CORRECT formulation (production code):")
print(f"   Min value: {C_correct.min():.6e}")
print(f"   Max value: {C_correct.max():.6e}")
print(f"   Profile at t=1: C(0)={C_correct[0,-1]:.6f}, C(0.5)={C_correct[Nx//2,-1]:.6f}, C(1)={C_correct[-1,-1]:.6f}")

print("\n❌ BUGGY formulation (deprecated prototype):")
print(f"   Min value: {C_buggy.min():.6e}")
print(f"   Max value: {C_buggy.max():.6e}")
print(f"   Profile at t=1: C(0)={C_buggy[0,-1]:.6f}, C(0.5)={C_buggy[Nx//2,-1]:.6f}, C(1)={C_buggy[-1,-1]:.6f}")

# Validation
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

if 0 <= C_correct.min() and C_correct.max() <= 1.5:
    print("✅ CORRECT formulation: physically admissible (C in [0, 1.5])")
else:
    print("❌ CORRECT formulation: physically inadmissible!")

if C_buggy.max() > 1e3 or C_buggy.min() < -1:
    print("❌ BUGGY formulation: diverges numerically (as expected)")
else:
    print("⚠️  BUGGY formulation: unexpected behavior")