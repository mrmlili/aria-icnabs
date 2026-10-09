import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def mittag_leffler(z, alpha, beta=1.0, max_terms=100, tol=1e-12):
    """محاسبه E_{alpha,beta}(z) با سری توانی."""
    z = float(z)
    result = 0.0
    for k in range(max_terms):
        term = z**k / gamma(alpha * k + beta)
        result += term
        if abs(term) < tol:
            break
    return result

# ==== تست ====
alpha = 0.8
mu = alpha / (1 - alpha)

def E_alpha(z):
    return mittag_leffler(z, alpha, beta=1.0)

dt = 0.01
t_n = 1.0

print("=" * 70)
print("مقایسه وزن ABC با وزن Caputo L1")
print("=" * 70)
print(f"{'k':>4} | {'w_ABC':>16} | {'w_Caputo L1':>16} | {'نسبت':>8}")
print("-" * 60)

for k in [0, 10, 50, 99]:
    t_k = k * dt
    t_kp1 = (k + 1) * dt
    
    integrand = lambda s: E_alpha(-mu * (t_n - s)**alpha)
    w_ABC, _ = quad(integrand, t_k, t_kp1)
    
    w_Caputo = ((t_n - t_k)**(1-alpha) - (t_n - t_kp1)**(1-alpha)) / (1-alpha)
    
    ratio = w_ABC / w_Caputo if w_Caputo > 0 else float('nan')
    print(f"{k:>4} | {w_ABC:>16.6e} | {w_Caputo:>16.6e} | {ratio:>8.4f}")