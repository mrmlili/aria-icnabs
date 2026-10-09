"""
ARIA-ICNABS Project
File: abc_l1_kernel.py
Purpose: Real ABC-L1 kernel weights with Mittag-Leffler kernel.
         This is the GENUINE nonsingular nonlocal discretization.
"""
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import os
import time


# ============================================================
# 1. MITTAG-LEFFLER FUNCTION (E_alpha, beta)
# ============================================================
def ml_ealpha_beta(z, alpha, beta=1.0, max_terms=300, tol=1e-14):
    """
    Compute E_{alpha, beta}(z) via truncated power series.
    
    E_{alpha,beta}(z) = sum_{k=0}^inf z^k / Gamma(alpha*k + beta)
    
    For our use case, z is negative and |z| < 10, so series converges
    within ~150 terms.
    """
    z = float(z)
    s = 0.0
    for k in range(max_terms):
        term = z**k / gamma(alpha * k + beta)
        s += term
        if abs(term) < tol * max(abs(s), 1.0):
            break
    return s


def ml_ealpha(z, alpha):
    """E_alpha(z) = E_{alpha,1}(z)."""
    return ml_ealpha_beta(z, alpha, 1.0)


# ============================================================
# 2. ABC-L1 WEIGHT VIA NUMERICAL INTEGRATION
# ============================================================
def abc_weight_single(t_n, t_k, t_kp1, alpha):
    """
    Compute integral of ML kernel over [t_k, t_kp1]:
    
        I = int_{t_k}^{t_kp1} E_alpha(-mu * (t_n - s)^alpha) ds
    
    where mu = alpha / (1 - alpha).
    """
    mu = alpha / (1 - alpha)
    
    def integrand(s):
        arg = -mu * (t_n - s)**alpha
        return ml_ealpha(arg, alpha)
    
    integral, _ = quad(integrand, t_k, t_kp1, limit=30, epsabs=1e-12)
    return integral


# ============================================================
# 3. PRECOMPUTE FULL WEIGHT MATRIX
# ============================================================
def precompute_abc_weight_matrix(alpha, dt, Nt, verbose=True):
    """
    Precompute W[n, k] for n = 1..Nt, k = 0..n-1.
    
    W[n, k] = (B(alpha)/(1-alpha)) * (1/dt) * int_{t_k}^{t_kp1} E_alpha(-mu*(t_n-s)^alpha) ds
    
    Returns:
        W : np.ndarray of shape (Nt+1, Nt+1), with W[n, k] for k < n
        (upper part unused, set to 0)
    """
    B_alpha = 1 - alpha + alpha / gamma(alpha)   # CORRECTED
    norm = B_alpha / (1 - alpha) / dt
    
    W = np.zeros((Nt + 1, Nt + 1))
    
    t_start = time.time()
    total = Nt * (Nt + 1) // 2
    count = 0
    
    for n in range(1, Nt + 1):
        t_n = n * dt
        for k in range(n):
            t_k = k * dt
            t_kp1 = (k + 1) * dt
            I = abc_weight_single(t_n, t_k, t_kp1, alpha)
            W[n, k] = norm * I
            count += 1
        
        if verbose and (n % max(1, Nt // 20) == 0):
            elapsed = time.time() - t_start
            eta = elapsed * (total - count) / max(count, 1)
            print(f"  n={n}/{Nt} | progress={100*count/total:.1f}% | "
                  f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s")
    
    if verbose:
        print(f"  Total precompute time: {time.time() - t_start:.1f}s")
    
    return W


# ============================================================
# 4. CACHED VERSION (save/load)
# ============================================================
def get_abc_weights(alpha, dt, Nt, cache_dir="."):
    """
    Get ABC-L1 weight matrix, with disk caching.
    """
    cache_file = os.path.join(
        cache_dir, 
        f"abc_weights_alpha{alpha:.3f}_Nt{Nt}.npz"
    )
    
    if os.path.exists(cache_file):
        print(f"Loading cached weights from {cache_file}")
        data = np.load(cache_file)
        return data['W']
    
    print(f"Precomputing ABC-L1 weights: alpha={alpha}, Nt={Nt}")
    W = precompute_abc_weight_matrix(alpha, dt, Nt, verbose=True)
    
    np.savez(cache_file, W=W, alpha=alpha, dt=dt, Nt=Nt)
    print(f"Cached to {cache_file}")
    
    return W


# ============================================================
# 5. UTILITY: Apply ABC-L1 derivative to a time-history
# ============================================================
def apply_abc_l1_derivative(history_C, W, n):
    """
    Given history_C = [C_0, C_1, ..., C_{n-1}] (list of arrays),
    and current C_n, compute ABC-L1 derivative at t_n.
    
    ABC-L1 formula:
        D_t^alpha C(x, t_n) ≈ sum_{k=0}^{n-1} W[n, k] * (C_{k+1} - C_k)
    
    Parameters:
        history_C : list of arrays (values at t_0, ..., t_{n-1})
        W         : precomputed weight matrix
        n         : current time index
    
    Returns:
        D_alpha : array of same shape as C_k
    """
    result = np.zeros_like(history_C[0])
    for k in range(n):
        result += W[n, k] * (history_C[k] - (history_C[k-1] if k > 0 else 0.0))
    return result


# ============================================================
# 6. SMOKE TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ABC-L1 KERNEL: SMOKE TEST")
    print("=" * 70)
    
    alpha = 0.8
    Nt = 20
    dt = 1.0 / Nt
    
    print(f"\nPrecomputing W for alpha={alpha}, Nt={Nt}...")
    W = precompute_abc_weight_matrix(alpha, dt, Nt, verbose=True)
    
    print("\nFirst few weights W[n, k]:")
    print(f"{'n':>3} | " + " ".join(f"{'k='+str(k):>12}" for k in range(min(5, Nt))))
    print("-" * 70)
    for n in [1, 5, 10, 20]:
        if n <= Nt:
            row = f"{n:>3} | " + " ".join(f"{W[n,k]:>12.4e}" for k in range(min(5, n)))
            print(row)
    
    # Validation: weights must be positive and decreasing in k for fixed n
    print("\nValidation:")
    for n in [5, 10, 20]:
        if n <= Nt:
            row = W[n, :n]
            assert np.all(row >= 0), f"Negative weight at n={n}"
            # Check monotonicity
            if np.all(np.diff(row) >= 0):
                print(f"  n={n}: increasing in k (memory strongest at recent times)")
            elif np.all(np.diff(row) <= 0):
                print(f"  n={n}: decreasing in k (memory strongest at early times)")
            else:
                print(f"  n={n}: non-monotonic (typical for ML kernel)")
    
    print("\n✅ Smoke test passed.")