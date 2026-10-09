"""
ARIA-ICNABS Project
File: 02b_Reference_ABC_Solver.py
Purpose: Reference solver with GENUINE ABC-L1 discretization
         (Mittag-Leffler kernel, no Caputo approximation)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.linalg import solve_banded
from abc_l1_kernel import get_abc_weights

# ============================================================
# PARAMETERS
# ============================================================
L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
alpha = 0.8
C0, Cin = 0.0, 1.0

Nx = 100
Nt = 200       # reduced for ABC-L1 (expensive)
dx = L / Nx
dt = T / Nt

x = np.linspace(0, L, Nx + 1)

# ============================================================
# SOLVER
# ============================================================
def solve_abc_l1_ade():
    """
    Solve time-fractional ADE with ABC-L1 discretization.
    
    Discretization:
        sum_{k=0}^{n-1} W[n,k] * (C_{k+1} - C_k) = L_x C_n
    
    where L_x is the spatial operator.
    """
    print("Loading ABC-L1 weights...")
    W = get_abc_weights(alpha, dt, Nt, cache_dir=".")
    print(f"W shape: {W.shape}, max = {W.max():.4e}, min = {W.min():.4e}")
    
    C = np.zeros((Nx + 1, Nt + 1))
    C[:, 0] = C0
    C[0, :] = Cin
    
    D = D0 * (1 + beta * x)
    
    # Tridiagonal coefficients (time-independent)
    a = np.zeros(Nx - 1)
    b = np.zeros(Nx - 1)
    c = np.zeros(Nx - 1)
    
    for i in range(1, Nx):
        a[i-1] = -(D[i] / dx**2 + v / (2*dx))
        b[i-1] = W[1, 0] + 2 * D[i] / dx**2      # will update per n
        c[i-1] = -(D[i] / dx**2 - v / (2*dx))
    
    # Time-stepping
    for n in range(1, Nt + 1):
        # Build RHS
        d = np.zeros(Nx - 1)
        for i in range(1, Nx):
            rhs = 0.0
            for k in range(n):
                rhs += W[n, k] * C[i, k+1] if k + 1 <= n - 1 else 0.0
                rhs -= W[n, k] * C[i, k]
            d[i-1] = -rhs
        
        # Update diagonal with W[n, n-1]
        b_now = W[n, n-1] + 2 * D[1:Nx] / dx**2
        
        # Boundary contributions
        d[0] += -a[0] * C[0, n]
        
        # Solve tridiagonal
        ab = np.zeros((3, Nx - 1))
        ab[0, 1:] = c[:-1]
        ab[1, :] = b_now
        ab[2, :-1] = a[1:]
        
        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx - 1, n]
        
        if n % max(1, Nt // 10) == 0:
            print(f"  step {n}/{Nt}")
    
    return x, C


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REFERENCE SOLVER: ABC-L1 (Mittag-Leffler kernel)")
    print("=" * 60)
    
    x, C = solve_abc_l1_ade()
    
    # Save
    np.savez("reference_abc.npz", x=x, C=C, alpha=alpha, Nt=Nt)
    print("\n✅ Reference data saved to reference_abc.npz")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, C[:, -1], 'r-', linewidth=2.5, label=f'ABC-L1 (alpha={alpha})')
    ax.set_xlabel("x")
    ax.set_ylabel(f"C(x, t={T})")
    ax.set_title("Reference solution: ABC-L1 with ML kernel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reference_abc.png", dpi=150)
    plt.show()