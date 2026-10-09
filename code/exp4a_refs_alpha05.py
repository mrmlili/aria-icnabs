"""
Experiment 4a: Generate three references for alpha = 0.5
"""
import time
import numpy as np
from scipy.special import gamma
from scipy.linalg import solve_banded
from scipy.integrate import quad

L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
alpha, C0, Cin = 0.5, 0.0, 1.0

B_alpha = 1 - alpha + alpha / gamma(alpha)
c_alpha = B_alpha / (1 - alpha)
print(f"alpha = {alpha}")
print(f"B(alpha) = {B_alpha:.6f}")
print(f"c(alpha) = {c_alpha:.6f}")


def solve_classical(Nx=200, Nt=2000):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    D_eff = (D0 / c_alpha) * (1 + beta * x)
    v_eff = v / c_alpha
    a = np.zeros(Nx - 1); b = np.zeros(Nx - 1); c = np.zeros(Nx - 1)
    for i in range(1, Nx):
        a[i-1] = -(dt * D_eff[i] / dx**2 + dt * v_eff / (2*dx))
        b[i-1] = 1 + 2 * dt * D_eff[i] / dx**2
        c[i-1] = -(dt * D_eff[i] / dx**2 - dt * v_eff / (2*dx))
    b[-1] += c[-1]
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
    return x, C


def caputo_weights(alpha, dt, Nt):
    B = 1 - alpha + alpha / gamma(alpha)
    scale = B / (1 - alpha) / gamma(2 - alpha) / (dt ** alpha)
    b = np.array([(k + 1)**(1 - alpha) - k**(1 - alpha) for k in range(Nt + 1)])
    return scale * b


def solve_caputo(Nx=100, Nt=500):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    D = D0 * (1 + beta * x)
    weights = caputo_weights(alpha, dt, Nt)
    a = np.zeros(Nx - 1); b = np.zeros(Nx - 1); c = np.zeros(Nx - 1)
    for i in range(1, Nx):
        a[i-1] = -(D[i] / dx**2 + v / (2*dx))
        b[i-1] = weights[0] + 2 * D[i] / dx**2
        c[i-1] = -(D[i] / dx**2 - v / (2*dx))
    b[-1] += c[-1]
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
    return x, C


def ml_ealpha(z, alpha, max_terms=300, tol=1e-14):
    z = float(z)
    s = 0.0
    for k in range(max_terms):
        term = z**k / gamma(alpha * k + 1)
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
            integrand = lambda s: ml_ealpha(-mu * (t_n - s)**alpha, alpha)
            integral, _ = quad(integrand, t_k, t_kp1, limit=30, epsabs=1e-12)
            W[n, k] = norm * integral
    return W


def solve_abc(Nx=100, Nt=200):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    D = D0 * (1 + beta * x)
    print(f"  Precomputing ABC weights (Nt={Nt})...")
    t0 = time.time()
    W = precompute_abc_weights(alpha, dt, Nt)
    print(f"  done in {time.time()-t0:.1f}s")
    C = np.zeros((Nx + 1, Nt + 1))
    C[:, 0] = C0
    C[0, :] = Cin
    for n in range(1, Nt + 1):
        d = np.zeros(Nx - 1)
        for i in range(1, Nx):
            rhs = 0.0
            for k in range(n):
                Ckp1 = C[i, k+1] if k + 1 <= n - 1 else 0.0
                rhs += W[n, k] * (Ckp1 - C[i, k])
            d[i-1] = -rhs
        a_i = np.array([-(D[i] / dx**2 + v / (2*dx)) for i in range(1, Nx)])
        b_i = np.array([W[n, n-1] + 2 * D[i] / dx**2 for i in range(1, Nx)])
        c_i = np.array([-(D[i] / dx**2 - v / (2*dx)) for i in range(1, Nx)])
        b_i[-1] += c_i[-1]
        ab = np.zeros((3, Nx - 1))
        ab[0, 1:] = c_i[:-1]; ab[1, :] = b_i; ab[2, :-1] = a_i[1:]
        d[0] += -a_i[0] * C[0, n]
        C[1:Nx, n] = solve_banded((1, 1), ab, d)
        C[Nx, n] = C[Nx-1, n]
    return x, C


if __name__ == "__main__":
    print("\n[1/3] Classical (alpha=0.5)")
    x, C = solve_classical()
    np.savez("reference_classical_alpha05.npz", x=x, C=C, alpha=alpha)
    print(f"  C(0,T)={C[0,-1]:.6f}, C(L/2,T)={C[50,-1]:.6f}, C(L,T)={C[100,-1]:.6f}")

    print("\n[2/3] Caputo-L1 (alpha=0.5)")
    x, C = solve_caputo()
    np.savez("reference_caputo_alpha05.npz", x=x, C=C, alpha=alpha)
    print(f"  C(0,T)={C[0,-1]:.6f}, C(L/2,T)={C[50,-1]:.6f}, C(L,T)={C[100,-1]:.6f}")

    print("\n[3/3] ABC-L1 (alpha=0.5)")
    x, C = solve_abc()
    np.savez("reference_abc_alpha05.npz", x=x, C=C, alpha=alpha)
    print(f"  C(0,T)={C[0,-1]:.6f}, C(L/2,T)={C[50,-1]:.6f}, C(L,T)={C[100,-1]:.6f}")

    print("\n✅ All three references saved")