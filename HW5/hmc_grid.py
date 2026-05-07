"""Self-contained HMC grid search over (sigma, epsilon, T).

Run: python hmc_grid.py
Saves: hmc_grid_best.png, hmc_grid_top_configs.png
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Target distribution (copied from hw5.py)
# ---------------------------------------------------------------------------
def gauss(x, mu, omega):
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))
    return (0.5 * omega / np.pi) * np.exp(-0.5 * omega * np.sum((x - mu) ** 2, axis=-1))


def grad_gauss(x, mu, omega):
    g = gauss(x, mu, omega)
    return (-0.5 * omega ** 2 / np.pi) * (
        g.reshape((g.shape[0], 1)).repeat(2, axis=1) * (x - mu)
    )


def phi(x):
    if len(x.shape) == 1 or x.shape[0] == 0:
        return np.column_stack((x[0], x[1] + 0.25 * np.sin(4.0 * np.pi * x[0])))
    return np.column_stack((x[:, 0], x[:, 1] + 0.25 * np.sin(4.0 * np.pi * x[:, 0])))


def dphi_transpose_dot_v(x, v):
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))
    return np.column_stack(
        (
            v[:, 0] + v[:, 1] * np.pi * np.cos(4.0 * np.pi * x[:, 0]),
            v[:, 1],
        )
    )


mu1 = np.array([1.0, 1.0])
mu2 = np.array([-1.0, -1.0])
omega1 = 9.0
omega2 = 9.0
lambda1 = 0.4
lambda2 = 0.6
EPS_LOG = 1.0e-8


def U(x):
    e1 = lambda1 * gauss(x, mu1, omega1)
    e2 = lambda2 * gauss(phi(x), mu2, omega2)
    return -np.log(e1 + e2 + EPS_LOG)


def grad_U(x):
    e1 = lambda1 * gauss(x, mu1, omega1)
    e2 = lambda2 * gauss(phi(x), mu2, omega2)
    grad_e1 = lambda1 * grad_gauss(x, mu1, omega1)
    grad_e2 = lambda2 * dphi_transpose_dot_v(x, grad_gauss(phi(x), mu2, omega2))
    a = (-1.0 / (e1 + e2 + EPS_LOG)).reshape((e1.shape[0], 1))
    return a.repeat(2, axis=1) * (grad_e1 + grad_e2)


def true_pdf(x):
    e1 = lambda1 * gauss(x, mu1, omega1)
    e2 = lambda2 * gauss(phi(x), mu2, omega2)
    return e1 + e2


# ---------------------------------------------------------------------------
# HMC sampler (corrected)
# ---------------------------------------------------------------------------
class HMC:
    def __init__(self, N=200, sigma=1.0, epsilon=0.05, T=1000):
        self.N = N
        self.sigma = sigma
        self.epsilon = epsilon
        self.T = T
        self.samples = np.zeros((N, 2))
        self.accepted = np.zeros(N)

    def leapfrog(self, current_q):
        eps = self.epsilon
        sigma2 = self.sigma ** 2

        q = current_q.copy()
        p = self.sigma * np.random.randn(2)
        current_p = p.copy()

        p = p - eps * grad_U(q).flatten() / 2
        for _ in range(self.T - 1):
            q = q + eps * p / sigma2
            p = p - eps * grad_U(q).flatten()
        q = q + eps * p / sigma2
        p = p - eps * grad_U(q).flatten() / 2
        p = -p

        cU = float(U(current_q).item())
        cK = float(np.dot(current_p, current_p) / (2 * sigma2))
        pU = float(U(q).item())
        pK = float(np.dot(p, p) / (2 * sigma2))

        alpha = np.exp(cU + cK - pU - pK)
        if np.random.uniform() < alpha:
            return q, 1
        return current_q, 0

    def sample(self):
        self.samples[0] = np.random.randn(2)
        for i in range(1, self.N):
            q, a = self.leapfrog(self.samples[i - 1])
            self.samples[i] = q
            self.accepted[i] = a


# ---------------------------------------------------------------------------
# KDE / MSE evaluation
# ---------------------------------------------------------------------------
def kde_pdf(x_eval, samples, omega_kernel):
    if len(x_eval.shape) == 1:
        x_eval = x_eval.reshape(1, -1)
    p = np.zeros(x_eval.shape[0])
    for i in range(samples.shape[0]):
        p += gauss(x_eval, samples[i], omega_kernel)
    return p / samples.shape[0]


eval_axis = np.linspace(-3.0, 3.0, 30)
xx_e, yy_e = np.meshgrid(eval_axis, eval_axis)
eval_points = np.column_stack((xx_e.flatten(), yy_e.flatten()))
true_pdf_vals = true_pdf(eval_points)

OMEGA_KERNEL = 16.0


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Grid search (large T)
    # ---------------------------------------------------------------------------
    sigma_grid = [0.5, 2.0, 8.0]
    epsilon_grid = [0.005, 0.05]
    T_grid = [500, 1000]
    N_SEARCH = 10000

    np.random.seed(0)
    results = []
    total = len(sigma_grid) * len(epsilon_grid) * len(T_grid)
    i = 0
    t_start = time.time()
    for sigma in sigma_grid:
        for eps in epsilon_grid:
            for T in T_grid:
                i += 1
                t0 = time.time()
                s = HMC(N=N_SEARCH, sigma=sigma, epsilon=eps, T=T)
                s.sample()
                kde_vals = kde_pdf(eval_points, s.samples, OMEGA_KERNEL)
                mse = float(np.mean((kde_vals - true_pdf_vals) ** 2))
                acc = 100 * s.accepted.sum() / s.N
                elapsed = time.time() - t0
                results.append({"sigma": sigma, "eps": eps, "T": T,
                                "accept": acc, "mse": mse, "samples": s.samples})
                print(f"[{i:>2}/{total}] sigma={sigma:>4} eps={eps:>5} T={T:>5}  "
                      f"accept={acc:5.1f}%  MSE={mse:.4e}  ({elapsed:.1f}s)")

    print(f"\nTotal grid search time: {time.time() - t_start:.1f}s")

    results_sorted = sorted(results, key=lambda r: r["mse"])
    print("\nAll configurations sorted by MSE:")
    print(f"{'sigma':>6} {'eps':>6} {'T':>6} {'accept%':>9} {'MSE':>14}")
    for r in results_sorted:
        print(f"{r['sigma']:>6} {r['eps']:>6} {r['T']:>6} "
              f"{r['accept']:>9.1f} {r['mse']:>14.6e}")

    viable = [r for r in results_sorted if r["accept"] >= 10.0]
    best = viable[0]
    print(f"\nBest viable: sigma={best['sigma']}, eps={best['eps']}, T={best['T']}, "
          f"accept={best['accept']:.1f}%, MSE={best['mse']:.6e}")


    # ---------------------------------------------------------------------------
    # Plot best config
    # ---------------------------------------------------------------------------
    xcoord = np.linspace(-3.0, 3.0, 201)
    ycoord = np.linspace(-3.0, 3.0, 201)
    xx, yy = np.meshgrid(xcoord, ycoord)
    coords = np.column_stack((xx.flatten(), yy.flatten()))
    dens = np.exp(-U(coords)).reshape(xx.shape)
    levels = np.linspace(np.max(dens) / 100, np.max(dens), 99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].set_facecolor("black")
    axes[0].contourf(xx, yy, dens, levels=levels, cmap="gist_heat")
    axes[0].scatter(best["samples"][:, 0], best["samples"][:, 1], s=3, c="cyan")
    axes[0].scatter(eval_points[:, 0], eval_points[:, 1], s=4, c="lime",
                    marker="+", linewidths=0.6, label="30x30 eval grid")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect("equal")
    axes[0].set_title(rf"Best HMC: $\sigma$={best['sigma']}, $\epsilon$={best['eps']}, "
                      f"T={best['T']}\naccept={best['accept']:.1f}%, MSE={best['mse']:.3e}")
    axes[0].grid(color="white", linestyle="--")
    axes[0].legend(loc="upper right")

    kde_grid = kde_pdf(coords, best["samples"], OMEGA_KERNEL).reshape(xx.shape)
    axes[1].set_facecolor("black")
    axes[1].contourf(xx, yy, kde_grid,
                     levels=np.linspace(kde_grid.max() / 100, kde_grid.max(), 99),
                     cmap="gist_heat")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"KDE pdf from best samples\nMSE on 30x30 grid: {best['mse']:.3e}")
    axes[1].grid(color="white", linestyle="--")
    fig.savefig("hmc_grid_best.png", dpi=120, bbox_inches="tight")
    print("Saved: hmc_grid_best.png")


    # ---------------------------------------------------------------------------
    # Plot top 4 viable configurations
    # ---------------------------------------------------------------------------
    fig_top, axes_top = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for ax, r in zip(axes_top.flatten(), viable[:4]):
        ax.set_facecolor("black")
        ax.contourf(xx, yy, dens, levels=levels, cmap="gist_heat")
        ax.scatter(r["samples"][:, 0], r["samples"][:, 1], s=3, c="cyan")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(rf"$\sigma$={r['sigma']}, $\epsilon$={r['eps']}, T={r['T']}"
                     f"\naccept={r['accept']:.1f}%, MSE={r['mse']:.3e}")
        ax.grid(color="white", linestyle="--")
    fig_top.savefig("hmc_grid_top_configs.png", dpi=120, bbox_inches="tight")
    print("Saved: hmc_grid_top_configs.png")
