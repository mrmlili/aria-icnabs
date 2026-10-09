"""
Experiment 3: Reference grids convergence.
For each reference, solve with a coarse grid and a fine grid, then
compare their predictions on the finest common grid.

Classical: (200, 2000) vs (400, 4000)
Caputo:    (100, 500)  vs (200, 1000)
ABC:       (100, 200)  vs (150, 400)
"""
import os
import time
import numpy as np
from scipy.special import gamma
from scipy.linalg import solve_banded
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ============================================================
# SHARED PARAMETERS
# ============================================================
L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
alpha, C0, Cin = 0.8, 0.0, 1.0

B_alpha = 1 - alpha + alpha / gamma(alpha)
c_alpha = B_alpha / (1 - alpha)
print(f"c(alpha) = {c_alpha:.6f}")


# ============================================================
# CLASSICAL SOLVER
# ============================================================
def solve_classical(Nx, Nt):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    D_eff = (D0 / c_alpha) * (1 + beta * x)
    v_eff = v / c_alpha

    a = np.zeros(Nx - 1); b = np.zeros(Nx - 1); c = np.zeros(Nx - 1)
    for i in range(1, Nx):
        a[i-1] = -(dt * D_eff[i] / dx**2 + dt * v_eff / (2*dx))
        b[i-1] = 1 + 2 * dt * D_eff[i] / dx**2
        c[i-1] = -(dt * D_eff[i] / dx**2 - dt * v_eff / (2*dx))
    b[-1] = b[-1] + c[-1]

    ab = np.zeros((3, Nx - 1))
    ab[0, 1:] = c[:-1]; ab[1, :] = b; ab[2, :-1] = a[1:]

    C = np.zeros((Nx + 1, Nt + 1))
    C[:, 0] = C0
    C[0, :] = Cin

    for n in range(1, Nt + 1):
        rhs = C[1:Nx, n-1].copy()
        rhs[0] += -a[0] * C[0, n]
        C[1:Nx, n] = solve_banded((1, 1), ab, rhs)
        C[Nx, n] = C[Nx-1, n]

    return x, C[:, -1]


# ============================================================
# CAPUTO-L1 SOLVER
# ============================================================
def caputo_weights(alpha, dt, Nt):
    B = 1 - alpha + alpha / gamma(alpha)
    scale = B / (1 - alpha) / gamma(2 - alpha) / (dt ** alpha)
    b = np.array([(k + 1)**(1 - alpha) - k**(1 - alpha) for k in range(Nt + 1)])
    return scale * b


def solve_caputo(Nx, Nt):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    D = D0 * (1 + beta * x)
    weights = caputo_weights(alpha, dt, Nt)

    a = np.zeros(Nx - 1); b = np.zeros(Nx - 1); c = np.zeros(Nx - 1)
    for i in range(1, Nx):
        a[i-1] = -(D[i] / dx**2 + v / (2*dx))
        b[i-1] = weights[0] + 2 * D[i] / dx**2
        c[i-1] = -(D[i] / dx**2 - v / (2*dx))
    b[-1] = b[-1] + c[-1]

    ab = np.zeros((3, Nx - 1))
    ab[0, 1:] = c[:-1]; ab[1, :] = b; ab[2, :-1] = a[1:]

    C = np.zeros((Nx + 1, Nt + 1))
    C[:, 0] = C0
    C[0, :] = Cin

    for n in range(1, Nt + 1):
        d = np.zeros(Nx - 1)
        for i in range(1, Nx):
            rhs = 0.0
            for k in range(1, n):
                rhs += (weights[k-1] - weights[k]) * C[i, n-k]
            rhs += weights[n-1] * C[i, 0]
            d[i-1] = rhs
        d[0] += -a[0] * C[0, n]
        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx-1, n]

    return x, C[:, -1]


# ============================================================
# ABC-L1 SOLVER
# ============================================================
def ml_ealpha_beta(z, alpha, beta=1.0, max_terms=300, tol=1e-14):
    z = float(z)
    s = 0.0
    for k in range(max_terms):
        term = z**k / gamma(alpha * k + beta)
        s += term
        if abs(term) < tol * max(abs(s), 1.0):
            break
    return s


def precompute_abc_weights(alpha, dt, Nt):
    B = 1 - alpha + alpha / gamma(alpha)
    mu = alpha / (1 - alpha)
    norm = B / (1 - alpha) / dt
    W = np.zeros((Nt + 1, Nt + 1))
    for n in range(1, Nt + 1):
        t_n = n * dt
        for k in range(n):
            t_k = k * dt
            t_kp1 = (k + 1) * dt
            integrand = lambda s: ml_ealpha_beta(-mu * (t_n - s)**alpha, alpha)
            integral, _ = quad(integrand, t_k, t_kp1, limit=30, epsabs=1e-12)
            W[n, k] = norm * integral
    return W


def solve_abc(Nx, Nt):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    D = D0 * (1 + beta * x)
    print(f"    Precomputing ABC weights for Nt={Nt}...")
    t0 = time.time()
    W = precompute_abc_weights(alpha, dt, Nt)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Time-stepping
    C = np.zeros((Nx + 1, Nt + 1))
    C[:, 0] = C0
    C[0, :] = Cin

    for n in range(1, Nt + 1):
        # RHS: sum_{k=0}^{n-1} W[n,k] * (C^{k+1} - C^k)
        d = np.zeros(Nx - 1)
        for i in range(1, Nx):
            rhs = 0.0
            for k in range(n):
                Ckp1 = C[i, k+1] if k + 1 <= n - 1 else 0.0
                rhs += W[n, k] * (Ckp1 - C[i, k])
            d[i-1] = -rhs

        # Tridiagonal coefficients
        a_i = np.array([-(D[i] / dx**2 + v / (2*dx)) for i in range(1, Nx)])
        b_i = np.array([W[n, n-1] + 2 * D[i] / dx**2 for i in range(1, Nx)])
        c_i = np.array([-(D[i] / dx**2 - v / (2*dx)) for i in range(1, Nx)])
        b_i[-1] = b_i[-1] + c_i[-1]

        ab = np.zeros((3, Nx - 1))
        ab[0, 1:] = c_i[:-1]; ab[1, :] = b_i; ab[2, :-1] = a_i[1:]

        d[0] += -a_i[0] * C[0, n]

        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx-1, n]

    return x, C[:, -1]


# ============================================================
# COMPARISON
# ============================================================
def compare(x_coarse, C_coarse, x_fine, C_fine):
    """Interp coarse onto fine x-grid, compute MSE."""
    f = interp1d(x_coarse, C_coarse, kind='linear')
    C_coarse_on_fine = f(x_fine)
    diff = C_coarse_on_fine - C_fine
    return float(np.mean(diff**2)), float(np.max(np.abs(diff)))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}

    print("\n" + "=" * 60)
    print("[1] Classical reference")
    print("=" * 60)
    x1, C1 = solve_classical(200, 2000)
    x2, C2 = solve_classical(400, 4000)
    mse, mx = compare(x1, C1, x2, C2)
    print(f"  (200,2000) vs (400,4000): MSE = {mse:.4e}, Max = {mx:.4e}")
    results['classical'] = {'mse': mse, 'max': mx}

    print("\n" + "=" * 60)
    print("[2] Caputo-L1 reference")
    print("=" * 60)
    x1, C1 = solve_caputo(100, 500)
    x2, C2 = solve_caputo(200, 1000)
    mse, mx = compare(x1, C1, x2, C2)
    print(f"  (100,500) vs (200,1000): MSE = {mse:.4e}, Max = {mx:.4e}")
    results['caputo'] = {'mse': mse, 'max': mx}

    print("\n" + "=" * 60)
    print("[3] ABC-L1 reference")
    print("=" * 60)
    x1, C1 = solve_abc(100, 200)
    x2, C2 = solve_abc(150, 400)
    mse, mx = compare(x1, C1, x2, C2)
    print(f"  (100,200) vs (150,400): MSE = {mse:.4e}, Max = {mx:.4e}")
    results['abc'] = {'mse': mse, 'max': mx}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Reference grid convergence")
    print("=" * 60)
    print(f"{'Reference':<12} | {'MSE between grids':>18} | {'Max diff':>12}")
    print("-" * 50)
    for k, v in results.items():
        print(f"{k:<12} | {v['mse']:>18.4e} | {v['max']:>12.4e}")

    np.savez("exp3_reference_grids.npz",
             classical_mse=results['classical']['mse'],
             classical_max=results['classical']['max'],
             caputo_mse=results['caputo']['mse'],
             caputo_max=results['caputo']['max'],
             abc_mse=results['abc']['mse'],
             abc_max=results['abc']['max'])
    print("\nSaved: exp3_reference_grids.npz")