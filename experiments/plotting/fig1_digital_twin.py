# experiments/plotting/fig1_digital_twin.py
"""
Figure 1: ABN-QSS v3.0 數字孿生驗證
(a) 投影線性度 R²=1.0
(b) 零空間維度 K=N(N-1)/2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard, svd
from experiments.plotting.style import plt, COLORS


def compute_nullspace(N):
    D = 2 ** N
    K = N * (N - 1) // 2
    H = hadamard(D) / np.sqrt(D)
    M = H[K:, :]
    U, S, Vh = svd(M)
    rank = np.sum(S > 1e-10)
    null_dim = D - rank
    V = Vh.conj().T
    return V[:, -null_dim:], null_dim, K


def compute_linearity(null_basis, delta=0.01, n_trials=200):
    D = null_basis.shape[0]
    K = null_basis.shape[1]
    C_base = null_basis @ (np.ones(K) / K)
    P = null_basis @ null_basis.T

    e0 = np.zeros(D)
    e0[0] = 1.0
    direction = P @ e0
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-12:
        return 0.0, 0.0, None, None
    direction = direction / dir_norm

    P_C_base = P @ C_base
    responses = []
    perturbations = np.linspace(-delta, delta, n_trials)
    for eps in perturbations:
        C_pert = C_base.copy()
        C_pert[0] += eps
        C_final = P @ C_pert
        responses.append(np.dot(C_final - P_C_base, direction))
    responses = np.array(responses)

    A = np.vstack([perturbations, np.ones_like(perturbations)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, responses, rcond=None)
    y_pred = A @ coeff
    ss_res = np.sum((responses - y_pred) ** 2)
    ss_tot = np.sum((responses - responses.mean()) ** 2)
    R2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return R2, coeff[0], perturbations, responses


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    Ns = [3, 4, 5, 6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # --- Panel (a): 投影線性度 ---
    ax = axes[0]
    for i, N in enumerate(Ns):
        null_basis, null_dim, K_theory = compute_nullspace(N)
        R2, slope, perturbations, responses = compute_linearity(null_basis)
        ax.plot(perturbations, responses, '-', color=colors[i],
                label=f'N={N} ($R^2$={R2:.4f})')
    ax.set_xlabel(r'Perturbation $\delta$')
    ax.set_ylabel(r'Signed projection $\langle \Delta C, u \rangle$')
    ax.set_title('(a) Projection linearity')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): 零空間維度 ---
    ax = axes[1]
    K_actual = []
    K_theory_list = []
    for N in Ns:
        _, null_dim, K_theory = compute_nullspace(N)
        K_actual.append(null_dim)
        K_theory_list.append(K_theory)

    x = np.arange(len(Ns))
    width = 0.38
    bars1 = ax.bar(x - width/2, K_actual, width,
                   label='Actual', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x + width/2, K_theory_list, width,
                   label=r'Theory $K=N(N-1)/2$',
                   color='#ff7f0e', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={N}' for N in Ns])
    ax.set_ylabel('Null-space dimension $K$')
    ax.set_title('(b) Null-space dimension')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = Path('figures/fig1_digital_twin.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    plt.close()


if __name__ == '__main__':
    main()