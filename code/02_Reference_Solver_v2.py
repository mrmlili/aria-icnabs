"""
ARIA-ICNABS Project
File: 02_Reference_Solver_v2.py
Purpose: Caputo-L1 reference with corrected B(alpha) and explicit labeling.
         This is the "classical nonlocal" reference, distinct from ABC-L1.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.linalg import solve_banded

# ============================================================
# PARAMETERS
# ============================================================
L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
alpha = 0.8
C0, Cin = 0.0, 1.0

Nx = 100
Nt = 500    # closer to original 1000 but faster

dx = L / Nx
dt = T / Nt
x = np.linspace(0, L, Nx + 1)

# ============================================================
# CAPUTO-L1 WEIGHTS (with prefactor)
# ============================================================
def caputo_l1_weights(alpha, dt, Nt):
    """
    Caputo L1 with B(alpha)/(1-alpha) prefactor.
    b_k = (k+1)^(1-alpha) - k^(1-alpha)
    """
    B_alpha = 1 - alpha + alpha / gamma(alpha)   # CORRECTED
    scaling = B_alpha / (1 - alpha) / gamma(2 - alpha) / (dt ** alpha)
    b = np.array([(k + 1)**(1 - alpha) - k**(1 - alpha) for k in range(Nt + 1)])
    return scaling * b


# ============================================================
# SOLVER
# ============================================================
def solve_caputo_l1_ade():
    C = np.zeros((Nx + 1, Nt + 1))
    C[:, 0] = C0
    C[0, :] = Cin

    weights = caputo_l1_weights(alpha, dt, Nt)
    D = D0 * (1 + beta * x)

    a = np.zeros(Nx - 1)
    b = np.zeros(Nx - 1)
    c = np.zeros(Nx - 1)

    for i in range(1, Nx):
        a[i - 1] = -(D[i] / dx**2 + v / (2 * dx))
        b[i - 1] = weights[0] + 2 * D[i] / dx**2
        c[i - 1] = -(D[i] / dx**2 - v / (2 * dx))

    b[-1] = b[-1] + c[-1]

    ab = np.zeros((3, Nx - 1))
    ab[0, 1:] = c[:-1]
    ab[1, :] = b
    ab[2, :-1] = a[1:]

    for n in range(1, Nt + 1):
        d = np.zeros(Nx - 1)
        for i in range(1, Nx):
            rhs = 0.0
            for k in range(1, n):
                rhs += (weights[k - 1] - weights[k]) * C[i, n - k]
            rhs += weights[n - 1] * C[i, 0]
            d[i - 1] = rhs
        d[0] += -a[0] * C[0, n]
        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx - 1, n]

    return x, C


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CAPUTO-L1 REFERENCE SOLVER (corrected B(alpha))")
    print("=" * 60)
    print(f"alpha = {alpha}, Nt = {Nt}")
    B_val = 1 - alpha + alpha / gamma(alpha)
    print(f"B(alpha) = {B_val:.6f}")
    print(f"c(alpha) = B(alpha)/(1-alpha) = {B_val/(1-alpha):.6f}")

    x, C = solve_caputo_l1_ade()
    print(f"\nSolution shape: {C.shape}")
    print(f"  Min = {C.min():.6e}")
    print(f"  Max = {C.max():.6e}")
    print(f"  C(0, T)  = {C[0, -1]:.6f}")
    print(f"  C(L/2, T) = {C[Nx//2, -1]:.6f}")
    print(f"  C(L, T)  = {C[Nx, -1]:.6f}")

    np.savez("reference_caputo.npz", x=x, C=C, alpha=alpha, Nt=Nt,
             B_alpha=B_val, c_alpha=B_val/(1-alpha))
    print("\n✅ Saved: reference_caputo.npz")

    # Plot comparison with ABC
    try:
        data_abc = np.load("reference_abc.npz")
        x_abc = data_abc['x']
        C_abc = data_abc['C'][:, -1]
        has_abc = True
    except FileNotFoundError:
        has_abc = False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, C[:, -1], 'b-', lw=2.5, label=f'Caputo L1 (corrected)')
    if has_abc:
        ax.plot(x_abc, C_abc, 'r--', lw=2.5, label=f'ABC L1 (ML kernel)')
    ax.set_xlabel("x"); ax.set_ylabel("C(x, t=1)")
    ax.set_title(f"Reference comparison (alpha={alpha})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reference_comparison.png", dpi=150)
    plt.show()
    print("✅ Saved: reference_comparison.png")

    if has_abc:
        C_abc_interp = np.interp(x, x_abc, C_abc)
        diff = C_abc_interp - C[:, -1]
        mse = np.mean(diff**2)
        print(f"\nMSE(Caputo, ABC) = {mse:.6e}")
        print(f"Max |diff|        = {np.max(np.abs(diff)):.6e}")
        if mse > 1e-4:
            print("✅ The two references differ significantly.")
        else:
            print("⚠️  References are very close — investigate.")