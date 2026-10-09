"""
ARIA-ICNABS
File: 00_Classical_Reference.py
Purpose: Classical reference for the naive FD PINN.
         Solves c(alpha) * dC/dt = d_x(D d_x C) - v d_x C on [0,1].
"""
import numpy as np
from scipy.special import gamma
from scipy.linalg import solve_banded

# ---- Parameters (match PINN) ----
L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
alpha = 0.8
C0, Cin = 0.0, 1.0

Nx, Nt = 200, 2000
dx, dt = L/Nx, T/Nt
x = np.linspace(0, L, Nx+1)

B_alpha = 1 - alpha + alpha/gamma(alpha)
c_alpha = B_alpha / (1 - alpha)
print(f"B(alpha) = {B_alpha:.6f}, c(alpha) = {c_alpha:.6f}")

# Effective coefficients
D_eff = D0 / c_alpha
v_eff = v / c_alpha

D = D_eff * (1 + beta * x)

# Implicit Euler: (I - dt * L_x) C^{n+1} = C^n
a = np.zeros(Nx-1); b = np.zeros(Nx-1); c = np.zeros(Nx-1)
for i in range(1, Nx):
    a[i-1] = -(dt * D[i] / dx**2 + dt * v_eff / (2*dx))
    b[i-1] = 1 + 2 * dt * D[i] / dx**2
    c[i-1] = -(dt * D[i] / dx**2 - dt * v_eff / (2*dx))
b[-1] = b[-1] + c[-1]

ab = np.zeros((3, Nx-1))
ab[0, 1:] = c[:-1]; ab[1, :] = b; ab[2, :-1] = a[1:]

C = np.zeros((Nx+1, Nt+1))
C[:, 0] = C0
C[0, :] = Cin

for n in range(1, Nt+1):
    rhs = C[1:Nx, n-1].copy()
    rhs[0] += -a[0] * C[0, n]
    C[1:Nx, n] = solve_banded((1, 1), ab, rhs)
    C[Nx, n] = C[Nx-1, n]

np.savez("reference_classical.npz", x=x, C=C, c_alpha=c_alpha)
print(f"\nSaved reference_classical.npz")
print(f"C(0,T)={C[0,-1]:.6f}, C(L/2,T)={C[Nx//2,-1]:.6f}, C(L,T)={C[Nx,-1]:.6f}")