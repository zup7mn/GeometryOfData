"""Run top-4 HMC configs 5 times each, compare averages, visualize.

Saves: hmc_replicates.png
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmc_grid import (
    HMC, U, kde_pdf,
    eval_points, true_pdf_vals, OMEGA_KERNEL,
)


# Top 4 configs from the N=10000 grid search
configs = [
    {"sigma": 0.5, "eps": 0.005, "T": 1000},
    {"sigma": 2.0, "eps": 0.05,  "T": 1000},
    {"sigma": 2.0, "eps": 0.005, "T": 1000},
    {"sigma": 8.0, "eps": 0.05,  "T": 500},
]
N_REPS = 5
N_SAMPLES = 10_000


def run_one(cfg, seed):
    np.random.seed(seed)
    s = HMC(N=N_SAMPLES, sigma=cfg["sigma"], epsilon=cfg["eps"], T=cfg["T"])
    s.sample()
    kde_vals = kde_pdf(eval_points, s.samples, OMEGA_KERNEL)
    mse = float(np.mean((kde_vals - true_pdf_vals) ** 2))
    acc = 100 * s.accepted.sum() / s.N
    return s.samples, mse, acc


results = [[] for _ in configs]
t_start = time.time()
for c_idx, cfg in enumerate(configs):
    for r_idx in range(N_REPS):
        seed = c_idx * 100 + r_idx + 1
        t0 = time.time()
        samples, mse, acc = run_one(cfg, seed)
        results[c_idx].append((samples, mse, acc))
        dt = time.time() - t0
        print(f"cfg {c_idx+1}/{len(configs)} (sigma={cfg['sigma']}, eps={cfg['eps']}, "
              f"T={cfg['T']}) rep {r_idx+1}/{N_REPS}: "
              f"MSE={mse:.4e} accept={acc:.1f}% ({dt:.1f}s)")

print(f"\nTotal time: {time.time() - t_start:.1f}s")

# Per-config statistics
print(f"\n{'sigma':>6} {'eps':>6} {'T':>5} | "
      f"{'mean MSE':>12} {'std MSE':>12} {'min MSE':>12} | {'mean acc%':>10}")
print("-" * 80)
stats = []
for cfg, runs in zip(configs, results):
    mses = np.array([r[1] for r in runs])
    accs = np.array([r[2] for r in runs])
    stats.append({
        "cfg": cfg,
        "mean_mse": float(mses.mean()),
        "std_mse": float(mses.std()),
        "min_mse": float(mses.min()),
        "mean_acc": float(accs.mean()),
    })
    print(f"{cfg['sigma']:>6} {cfg['eps']:>6} {cfg['T']:>5} | "
          f"{mses.mean():>12.4e} {mses.std():>12.4e} {mses.min():>12.4e} | "
          f"{accs.mean():>10.1f}")

best = min(stats, key=lambda s: s["mean_mse"])
print(f"\nBest by mean MSE: sigma={best['cfg']['sigma']}, eps={best['cfg']['eps']}, "
      f"T={best['cfg']['T']}, mean MSE = {best['mean_mse']:.4e}")


# 4 rows (configs) x 5 cols (replicates)
xcoord = np.linspace(-3.0, 3.0, 201)
ycoord = np.linspace(-3.0, 3.0, 201)
xx, yy = np.meshgrid(xcoord, ycoord)
coords = np.column_stack((xx.flatten(), yy.flatten()))
dens = np.exp(-U(coords)).reshape(xx.shape)
levels = np.linspace(np.max(dens) / 100, np.max(dens), 99)

fig, axes = plt.subplots(len(configs), N_REPS,
                         figsize=(4 * N_REPS, 4 * len(configs)),
                         constrained_layout=True)
for c_idx, (cfg, runs) in enumerate(zip(configs, results)):
    s = stats[c_idx]
    for r_idx, (samples, mse, acc) in enumerate(runs):
        ax = axes[c_idx, r_idx]
        ax.set_facecolor("black")
        ax.contourf(xx, yy, dens, levels=levels, cmap="gist_heat")
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c="cyan", alpha=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        if r_idx == 0:
            ax.set_ylabel(rf"$\sigma$={cfg['sigma']}, $\epsilon$={cfg['eps']}, T={cfg['T']}"
                          + "\n"
                          + rf"mean MSE={s['mean_mse']:.3e}",
                          fontsize=10)
        ax.set_title(f"rep {r_idx+1}: MSE={mse:.3e}, acc={acc:.1f}%", fontsize=9)
        ax.grid(color="white", linestyle="--", alpha=0.3)

fig.suptitle("Top-4 HMC configs, 5 replicates each (N=10000 samples per run)",
             fontsize=14)
fig.savefig("hmc_replicates.png", dpi=120, bbox_inches="tight")
print("Saved: hmc_replicates.png")
