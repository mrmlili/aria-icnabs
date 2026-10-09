"""
Experiment 4b: Train three PINNs for alpha = 0.5, 3 seeds each.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma
from scipy.integrate import quad

# ============================================================
# CONFIG
# ============================================================
class Config:
    L, T = 1.0, 1.0
    D0, beta, v = 0.01, 0.5, 0.1
    alpha, C0, Cin = 0.5, 0.0, 1.0
    n_hidden, n_neurons = 4, 64
    n_epochs, lr = 8000, 1e-3
    n_ic, n_bc = 300, 300

config = Config()
B_alpha = 1 - config.alpha + config.alpha / gamma(config.alpha)
c_alpha = B_alpha / (1 - config.alpha)
print(f"c(alpha) = {c_alpha:.6f}")


class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = [nn.Linear(2, cfg.n_neurons), nn.Tanh()]
        for _ in range(cfg.n_hidden - 1):
            layers += [nn.Linear(cfg.n_neurons, cfg.n_neurons), nn.Tanh()]
        layers.append(nn.Linear(cfg.n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ============================================================
# NAIVE FD
# ============================================================
def loss_naive(model, cfg):
    x_pde = torch.rand(1000, 1) * cfg.L
    t_pde = torch.rand(1000, 1) * cfg.T
    x_pde.requires_grad_(True); t_pde.requires_grad_(True)
    C = model(x_pde, t_pde)
    dC_dx = torch.autograd.grad(C, x_pde, torch.ones_like(C), create_graph=True)[0]
    d2C_dx2 = torch.autograd.grad(dC_dx, x_pde, torch.ones_like(dC_dx), create_graph=True)[0]
    D_x = cfg.D0 * (1 + cfg.beta * x_pde)
    diff = cfg.D0 * cfg.beta * dC_dx + D_x * d2C_dx2
    adv = cfg.v * dC_dx
    C_t = model(x_pde, t_pde)
    C_tm = model(x_pde, torch.clamp(t_pde - 0.01, min=0))
    frac = c_alpha * (C_t - C_tm) / 0.01
    Lp = torch.mean((frac - diff + adv)**2)
    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    Li = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - cfg.C0)**2)
    t_b1 = torch.rand(cfg.n_bc, 1) * cfg.T
    Lb1 = torch.mean((model(torch.zeros_like(t_b1), t_b1) - cfg.Cin)**2)
    x_b2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_b2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_b2.requires_grad_(True)
    Cb = model(x_b2, t_b2)
    dCb = torch.autograd.grad(Cb, x_b2, torch.ones_like(Cb), create_graph=True)[0]
    Lb2 = torch.mean(dCb**2)
    return Lp + 10 * Li + 10 * Lb1 + 10 * Lb2


# ============================================================
# CAPUTO L1
# ============================================================
Nt_hist = 50
Nx_colloc = 60


def caputo_weights(alpha, dt, Nt):
    B = 1 - alpha + alpha / gamma(alpha)
    scale = B / (1 - alpha) / gamma(2 - alpha) / (dt ** alpha)
    b = np.array([(k + 1)**(1 - alpha) - k**(1 - alpha) for k in range(Nt + 1)])
    return scale * b


def loss_caputo(model, cfg, W):
    x_pde = torch.rand(Nx_colloc, 1) * cfg.L
    tg = torch.linspace(0, cfg.T, Nt_hist + 1)
    hist = [model(x_pde, tg[k].reshape(1, 1).expand(Nx_colloc, 1)) for k in range(Nt_hist + 1)]
    Lp = 0.0
    for n in range(1, Nt_hist + 1):
        frac = W[0] * (hist[n] - hist[n-1])
        for k in range(1, n):
            frac = frac + W[k] * (hist[n-k] - hist[n-k-1])
        x_ad = x_pde.clone().requires_grad_(True)
        t_n = tg[n].reshape(1, 1).expand(Nx_colloc, 1)
        t_ad = t_n.clone().requires_grad_(True)
        C = model(x_ad, t_ad)
        dC = torch.autograd.grad(C, x_ad, torch.ones_like(C), create_graph=True)[0]
        d2C = torch.autograd.grad(dC, x_ad, torch.ones_like(dC), create_graph=True)[0]
        D_x = cfg.D0 * (1 + cfg.beta * x_ad)
        diff = cfg.D0 * cfg.beta * dC + D_x * d2C
        adv = cfg.v * dC
        Lp = Lp + torch.mean((frac - diff + adv)**2)
    Lp = Lp / Nt_hist
    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    Li = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - cfg.C0)**2)
    t_b1 = torch.rand(cfg.n_bc, 1) * cfg.T
    Lb1 = torch.mean((model(torch.zeros_like(t_b1), t_b1) - cfg.Cin)**2)
    x_b2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_b2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_b2.requires_grad_(True)
    Cb = model(x_b2, t_b2)
    dCb = torch.autograd.grad(Cb, x_b2, torch.ones_like(Cb), create_graph=True)[0]
    Lb2 = torch.mean(dCb**2)
    return Lp + 10 * Li + 10 * Lb1 + 10 * Lb2


# ============================================================
# ABC L1 (with precomputed W)
# ============================================================
def ml_ealpha_scipy(z, alpha, max_terms=300):
    z = float(z)
    s = 0.0
    for k in range(max_terms):
        term = z**k / gamma(alpha * k + 1)
        s += term
        if abs(term) < 1e-14 * max(abs(s), 1.0):
            break
    return s


def get_abc_weights(alpha, dt, Nt):
    cache = f"abc_weights_alpha{alpha:.3f}_Nt{Nt}.npz"
    if os.path.exists(cache):
        print(f"  Loading cached {cache}")
        return torch.tensor(np.load(cache)['W'], dtype=torch.float32)
    print(f"  Precomputing W for alpha={alpha}, Nt={Nt}...")
    B = 1 - alpha + alpha / gamma(alpha)
    mu = alpha / (1 - alpha)
    norm = B / (1 - alpha) / dt
    W = np.zeros((Nt + 1, Nt + 1))
    t0 = time.time()
    for n in range(1, Nt + 1):
        t_n = n * dt
        for k in range(n):
            t_k = k * dt
            t_kp1 = (k + 1) * dt
            integrand = lambda s: ml_ealpha_scipy(-mu * (t_n - s)**alpha, alpha)
            integral, _ = quad(integrand, t_k, t_kp1, limit=30, epsabs=1e-12)
            W[n, k] = norm * integral
    print(f"  done in {time.time()-t0:.1f}s")
    np.savez(cache, W=W)
    return torch.tensor(W, dtype=torch.float32)


def loss_abc(model, cfg, W):
    x_pde = torch.rand(Nx_colloc, 1) * cfg.L
    tg = torch.linspace(0, cfg.T, Nt_hist + 1)
    hist = [model(x_pde, tg[k].reshape(1, 1).expand(Nx_colloc, 1)) for k in range(Nt_hist + 1)]
    Lp = 0.0
    for n in range(1, Nt_hist + 1):
        frac = 0.0
        for k in range(n):
            frac = frac + W[n, k] * (hist[k+1] - hist[k])
        x_ad = x_pde.clone().requires_grad_(True)
        t_n = tg[n].reshape(1, 1).expand(Nx_colloc, 1)
        t_ad = t_n.clone().requires_grad_(True)
        C = model(x_ad, t_ad)
        dC = torch.autograd.grad(C, x_ad, torch.ones_like(C), create_graph=True)[0]
        d2C = torch.autograd.grad(dC, x_ad, torch.ones_like(dC), create_graph=True)[0]
        D_x = cfg.D0 * (1 + cfg.beta * x_ad)
        diff = cfg.D0 * cfg.beta * dC + D_x * d2C
        adv = cfg.v * dC
        Lp = Lp + torch.mean((frac - diff + adv)**2)
    Lp = Lp / Nt_hist
    x_ic = torch.rand(cfg.n_ic, 1) * cfg.L
    Li = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - cfg.C0)**2)
    t_b1 = torch.rand(cfg.n_bc, 1) * cfg.T
    Lb1 = torch.mean((model(torch.zeros_like(t_b1), t_b1) - cfg.Cin)**2)
    x_b2 = torch.ones(cfg.n_bc, 1) * cfg.L
    t_b2 = torch.rand(cfg.n_bc, 1) * cfg.T
    x_b2.requires_grad_(True)
    Cb = model(x_b2, t_b2)
    dCb = torch.autograd.grad(Cb, x_b2, torch.ones_like(Cb), create_graph=True)[0]
    Lb2 = torch.mean(dCb**2)
    return Lp + 10 * Li + 10 * Lb1 + 10 * Lb2


# ============================================================
# TRAINING
# ============================================================
def train_one(method, seed, W=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)
    t0 = time.time()
    for ep in range(config.n_epochs):
        opt.zero_grad()
        if method == 'naivefd':
            loss = loss_naive(model, config)
        elif method == 'caputo':
            loss = loss_caputo(model, config, W)
        else:
            loss = loss_abc(model, config, W)
        loss.backward(); opt.step(); sch.step()
        if (ep+1) % 2000 == 0:
            print(f"    seed={seed} ep={ep+1} loss={loss.item():.4e}")
    return model, time.time() - t0


def evaluate(model, ref_file):
    d = np.load(ref_file)
    x_r = d['x']; C_r = d['C'][:, -1]
    x_t = torch.tensor(x_r, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)
    with torch.no_grad():
        C_p = model(x_t, t_t).numpy().flatten()
    mse = float(np.mean((C_p - C_r)**2))
    rmse = float(np.sqrt(mse))
    mx = float(np.max(np.abs(C_p - C_r)))
    return mse, rmse, mx


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    seeds = [42, 123, 7]

    # Prepare weights
    dt_cap = config.T / Nt_hist
    W_cap = torch.tensor(caputo_weights(config.alpha, dt_cap, Nt_hist), dtype=torch.float32)
    W_abc = get_abc_weights(config.alpha, dt_cap, Nt_hist)

    all_results = {}

    for method, W, ref_file in [
        ('naivefd', None, 'reference_classical_alpha05.npz'),
        ('caputo', W_cap, 'reference_caputo_alpha05.npz'),
        ('abc', W_abc, 'reference_abc_alpha05.npz'),
    ]:
        print(f"\n{'='*60}")
        print(f"Training {method.upper()} (alpha=0.5)")
        print(f"{'='*60}")
        mses, times = [], []
        for s in seeds:
            print(f"\n  --- seed {s} ---")
            model, el = train_one(method, s, W)
            mse, rmse, mx = evaluate(model, ref_file)
            print(f"    MSE={mse:.4e}  RMSE={rmse:.4e}  Max={mx:.4e}  Time={el:.0f}s")
            torch.save(model.state_dict(), f"pinn_{method}_alpha05_seed{s}.pth")
            mses.append(mse); times.append(el)
        all_results[method] = {
            'mse': np.array(mses),
            'time': np.array(times),
        }

    print("\n" + "=" * 70)
    print("SUMMARY: alpha = 0.5, 3 seeds each")
    print("=" * 70)
    print(f"{'Method':<12} | {'MSE (mean ± std)':<25} | {'Time (s)':<10}")
    print("-" * 70)
    for m in ['naivefd', 'caputo', 'abc']:
        mse_m = all_results[m]['mse'].mean()
        mse_s = all_results[m]['mse'].std()
        t_m = all_results[m]['time'].mean()
        print(f"{m:<12} | {mse_m:.4e} ± {mse_s:.4e}       | {t_m:.0f}")

    np.savez("exp4_alpha05_results.npz",
             naive_mse=all_results['naivefd']['mse'],
             naive_time=all_results['naivefd']['time'],
             caputo_mse=all_results['caputo']['mse'],
             caputo_time=all_results['caputo']['time'],
             abc_mse=all_results['abc']['mse'],
             abc_time=all_results['abc']['time'])
    print("\nSaved: exp4_alpha05_results.npz")