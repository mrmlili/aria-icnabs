"""
Base equation: ∂_τ C_1 = ∂_x(D ∂_x C_1) - v ∂_x C_1
with UNSCALED D0, v (c=1).
Solution on τ ∈ [0, 1] with Nt=2000.
"""
import numpy as np
from scipy.linalg import solve_banded

L, T = 1.0, 1.0
D0, beta, v = 0.01, 0.5, 0.1
C0, Cin = 0.0, 1.0

Nx, Nt = 200, 2000
dx, dt = L/Nx, T/Nt
x = np.linspace(0, L, Nx+1)

D = D0 * (1 + beta * x)

a = np.zeros(Nx-1); b = np.zeros(Nx-1); c = np.zeros(Nx-1)
for i in range(1, Nx):
    a[i-1] = -(dt * D[i] / dx**2 + dt * v / (2*dx))
    b[i-1] = 1 + 2 * dt * D[i] / dx**2
    c[i-1] = -(dt * D[i] / dx**2 - dt * v / (2*dx))
b[-1] = b[-1] + c[-1]

ab = np.zeros((3, Nx-1))
ab[0, 1:] = c[:-1]; ab[1, :] = b; ab[2, :-1] = a[1:]

C1 = np.zeros((Nx+1, Nt+1))
C1[:, 0] = C0
C1[0, :] = Cin

for n in range(1, Nt+1):
    rhs = C1[1:Nx, n-1].copy()
    rhs[0] += -a[0] * C1[0, n]
    C1[1:Nx, n] = solve_banded((1, 1), ab, rhs)
    C1[Nx, n] = C1[Nx-1, n]

np.savez("reference_base_c1.npz", x=x, t=np.linspace(0,T,Nt+1), C=C1)
print(f"✅ Saved reference_base_c1.npz")
print(f"Shape: {C1.shape}")
print(f"C(0,T)={C1[0,-1]:.6f}, C(L/2,T)={C1[Nx//2,-1]:.6f}, C(L,T)={C1[Nx,-1]:.6f}")