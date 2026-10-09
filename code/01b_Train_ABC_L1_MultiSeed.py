"""
ARIA-ICNABS Project
File: 01b_Train_ABC_L1_MultiSeed.py
Purpose: Run ABC-L1 PINN with seeds 123 and 7 (seed 42 already done).
         Aggregates all three seeds into a final summary table.
Note: This file is standalone — it does NOT import from 01_Train_PINN_ABC_L1.py.
      The original file is kept untouched.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIGURATION (identical to 01_Train_PINN_ABC_L1.py)
# ============================================================
class Config:
    L = 1.0
    T = 1.0
    D0 = 0.01
    beta = 0.5
    v = 0.1
    alpha = 0.8
    C0 = 0.0
    Cin = 1.0

    n_hidden = 4
    n_neurons = 64

    n_epochs = 8000
    lr = 1e-3
    n_ic = 300
    n_bc = 300

    n_time_steps = 50
    n_x_colloc = 60

config = Config()

# ============================================================
# PINN NETWORK (identical)
# ============================================================
class PINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [nn.Linear(2, config.n_neurons), nn.Tanh()]
        for _ in range(config.n_hidden - 1):
            layers += [nn.Linear(config.n_neurons, config.n_neurons), nn.Tanh()]
        layers.append(nn.Linear(config.n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# ============================================================
# LOAD ABC WEIGHTS
# ============================================================
def load_abc_weights(alpha, dt, Nt):
    cache = f"abc_weights_alpha{alpha:.3f}_Nt{Nt}.npz"
    if os.path.exists(cache):
        print(f"  Loading cached ABC weights: {cache}")
        return torch.tensor(np.load(cache)['W'], dtype=torch.float32)

    print(f"  Precomputing ABC weights: alpha={alpha}, Nt={Nt}")
    from abc_l1_kernel import precompute_abc_weight_matrix
    W = precompute_abc_weight_matrix(alpha, dt, Nt, verbose=True)
    np.savez(cache, W=W, alpha=alpha, dt=dt, Nt=Nt)
    return torch.tensor(W, dtype=torch.float32)

# ============================================================
# LOSS (identical)
# ============================================================
def compute_loss(model, config, W):
    Nt = config.n_time_steps
    x_pde = torch.rand(config.n_x_colloc, 1) * config.L
    time_grid = torch.linspace(0, config.T, Nt + 1)

    history_C = []
    for k in range(Nt + 1):
        t_k = time_grid[k].reshape(1, 1).expand(config.n_x_colloc, 1)
        history_C.append(model(x_pde, t_k))

    loss_pde = 0.0
    for n in range(1, Nt + 1):
        frac = 0.0
        for k in range(n):
            frac = frac + W[n, k] * (history_C[k + 1] - history_C[k])

        x_ad = x_pde.clone().requires_grad_(True)
        t_n = time_grid[n].reshape(1, 1).expand(config.n_x_colloc, 1)
        t_ad = t_n.clone().requires_grad_(True)
        C_ad = model(x_ad, t_ad)

        dC_dx = torch.autograd.grad(
            C_ad, x_ad, torch.ones_like(C_ad), create_graph=True
        )[0]
        d2C_dx2 = torch.autograd.grad(
            dC_dx, x_ad, torch.ones_like(dC_dx), create_graph=True
        )[0]

        D_x = config.D0 * (1 + config.beta * x_ad)
        dD_dx = config.D0 * config.beta
        diffusion = dD_dx * dC_dx + D_x * d2C_dx2
        advection = config.v * dC_dx

        residual = frac - diffusion + advection
        loss_pde = loss_pde + torch.mean(residual**2)

    loss_pde = loss_pde / Nt

    x_ic = torch.rand(config.n_ic, 1) * config.L
    t_ic = torch.zeros(config.n_ic, 1)
    loss_ic = torch.mean((model(x_ic, t_ic) - config.C0)**2)

    t_bc1 = torch.rand(config.n_bc, 1) * config.T
    x_bc1 = torch.zeros(config.n_bc, 1)
    loss_bc1 = torch.mean((model(x_bc1, t_bc1) - config.Cin)**2)

    x_bc2 = torch.ones(config.n_bc, 1) * config.L
    t_bc2 = torch.rand(config.n_bc, 1) * config.T
    x_bc2.requires_grad_(True)
    C_bc2 = model(x_bc2, t_bc2)
    dC_dx_bc2 = torch.autograd.grad(
        C_bc2, x_bc2, torch.ones_like(C_bc2), create_graph=True
    )[0]
    loss_bc2 = torch.mean(dC_dx_bc2**2)

    total = loss_pde + 10 * loss_ic + 10 * loss_bc1 + 10 * loss_bc2
    return total, loss_pde, loss_ic, loss_bc1, loss_bc2

# ============================================================
# TRAIN (with seed-based saving)
# ============================================================
def train_one_seed(seed, config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Training ABC-L1 PINN with seed = {seed}")
    print(f"{'='*60}")

    model = PINN(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params} parameters")

    dt = config.T / config.n_time_steps
    W = load_abc_weights(config.alpha, dt, config.n_time_steps)
    print(f"  ABC weights: shape={tuple(W.shape)}, max={W.max():.4e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs
    )

    losses = []
    t0 = time.time()
    for epoch in range(config.n_epochs):
        optimizer.zero_grad()
        total, lp, li, lb1, lb2 = compute_loss(model, config, W)
        total.backward()
        optimizer.step()
        scheduler.step()
        losses.append(total.item())

        if (epoch + 1) % 1000 == 0:
            print(f"    epoch {epoch+1:>5}/{config.n_epochs} | "
                  f"total={total.item():.4e} | PDE={lp.item():.4e} | "
                  f"IC={li.item():.4e} | BC1={lb1.item():.4e} | BC2={lb2.item():.4e}")

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    return model, losses, elapsed

# ============================================================
# EVALUATE (against ABC reference)
# ============================================================
def evaluate_one_seed(model):
    data = np.load("reference_abc.npz")
    x_ref = data['x']
    C_ref = data['C'][:, -1]

    x_t = torch.tensor(x_ref, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.ones_like(x_t)

    with torch.no_grad():
        C_pred = model(x_t, t_t).numpy().flatten()

    mse = float(np.mean((C_pred - C_ref)**2))
    rmse = float(np.sqrt(mse))
    mx = float(np.max(np.abs(C_pred - C_ref)))
    return mse, rmse, mx, C_pred, C_ref, x_ref

# ============================================================
# MAIN: run seeds 123 and 7, then aggregate with seed 42
# ============================================================
if __name__ == "__main__":
    seeds_to_run = [123, 7]

    new_results = []

    for seed in seeds_to_run:
        model, losses, elapsed = train_one_seed(seed, config)

        # Save model
        model_path = f"pinn_abc_l1_seed{seed}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: {model_path}")

        # Evaluate
        mse, rmse, mx, C_pred, C_ref, x_ref = evaluate_one_seed(model)
        print(f"  MSE={mse:.4e}  RMSE={rmse:.4e}  Max={mx:.4e}")

        # Save per-seed results
        result_path = f"abc_l1_results_seed{seed}.npz"
        np.savez(result_path,
                 mse=mse, rmse=rmse, max_err=mx,
                 C_pred=C_pred, C_ref=C_ref, x=x_ref,
                 losses=np.array(losses),
                 elapsed=elapsed, seed=seed)
        print(f"  Saved: {result_path}")

        new_results.append({
            'seed': seed,
            'mse': mse,
            'rmse': rmse,
            'max': mx,
            'time': elapsed,
        })

    # ============================================================
    # AGGREGATE with seed 42 (already computed)
    # ============================================================
    print("\n" + "=" * 60)
    print("AGGREGATING ALL SEEDS")
    print("=" * 60)

    all_results = []

    # Try to load existing seed=42 results
    seed42_file = "abc_l1_pinn_results.npz"
    if os.path.exists(seed42_file):
        d = np.load(seed42_file)
        seed42 = {
            'seed': 42,
            'mse': float(d['mse']),
            'rmse': float(d['rmse']),
            'max': float(d['max_err']),
            'time': float(d['elapsed']),
        }
        all_results.append(seed42)
        print(f"  Loaded seed=42 from {seed42_file}: "
              f"MSE={seed42['mse']:.4e}, time={seed42['time']:.1f}s")
    else:
        print(f"  ⚠️  {seed42_file} not found — seed 42 will be missing from aggregate")

    all_results.extend(new_results)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: ABC-L1 PINN (3 seeds)")
    print("=" * 60)
    print(f"{'seed':>6} | {'MSE':>12} | {'RMSE':>12} | {'Max Error':>12} | {'Time (s)':>10}")
    print("-" * 66)
    for r in all_results:
        print(f"{r['seed']:>6} | {r['mse']:>12.4e} | {r['rmse']:>12.4e} | "
              f"{r['max']:>12.4e} | {r['time']:>10.1f}")

    mses = np.array([r['mse'] for r in all_results])
    rmses = np.array([r['rmse'] for r in all_results])
    maxes = np.array([r['max'] for r in all_results])

    print("-" * 66)
    print(f"{'mean':>6} | {mses.mean():>12.4e} | {rmses.mean():>12.4e} | "
          f"{maxes.mean():>12.4e} |")
    print(f"{'std':>6} | {mses.std():>12.4e} | {rmses.std():>12.4e} | "
          f"{maxes.std():>12.4e} |")

    # Save aggregated
    np.savez("abc_l1_multi_seed_results.npz",
             seeds=[r['seed'] for r in all_results],
             mse=mses, rmse=rmses, max_err=maxes,
             times=[r['time'] for r in all_results])
    print("\n  Saved: abc_l1_multi_seed_results.npz")

    # ============================================================
    # FINAL THREE-WAY TABLE
    # ============================================================
    print("\n" + "=" * 78)
    print("FINAL THREE-WAY COMPARISON TABLE")
    print("=" * 78)
    print(f"{'Residual':>14} | {'MSE (mean ± std)':>24} | {'Time (s)':>10}")
    print("-" * 78)

    # Load naive FD and Caputo
    try:
        nfd = np.load("naivefd_results.npz")
        print(f"{'Naive FD':>14} | {nfd['mse'].mean():>10.4e} ± {nfd['mse'].std():>8.4e} | "
              f"{nfd['times'].mean():>10.1f}")
    except FileNotFoundError:
        print(f"{'Naive FD':>14} | (naivefd_results.npz not found)")

    try:
        cap = np.load("caputo_l1_results.npz")
        print(f"{'Caputo-L1':>14} | {cap['mse'].mean():>10.4e} ± {cap['mse'].std():>8.4e} | "
              f"{cap['times'].mean():>10.1f}")
    except FileNotFoundError:
        print(f"{'Caputo-L1':>14} | (caputo_l1_results.npz not found)")

    print(f"{'ABC-L1':>14} | {mses.mean():>10.4e} ± {mses.std():>8.4e} | "
          f"{np.array([r['time'] for r in all_results]).mean():>10.1f}")

    print("=" * 78)
    print("\n✅ Multi-seed ABC-L1 run complete.")