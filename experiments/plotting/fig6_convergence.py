# experiments/plotting/fig6_convergence.py
"""
Figure 6: ZSP 收斂行為
(a) N=4 損失曲線（含前期震盪）
(b) 梯度範數衰減
(c) 失配度 D 演化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from core.constraint import MOTConstraintBuilder
from core.nullspace import NullSpaceExtractor
from core.solver import ProjectedGradientSolver
from experiments.plotting.style import plt, COLORS


def run_one(T=5, D=5, seed=42, max_iter=300):
    np.random.seed(seed)
    p = np.random.rand(T, D)
    builder = MOTConstraintBuilder(T, D, p, tau=0.1)
    M, _, _ = builder.build()
    extractor = NullSpaceExtractor(M)
    solver = ProjectedGradientSolver(
        M, extractor.null_basis, p, T, D,
        eta=0.005, gamma=0.2, D_th=0.02,
        max_iter=max_iter, min_iter=20
    )
    return solver.solve()


def main():
    result = run_one()
    losses = np.array(result['loss'])
    grad_norms = np.array(result['grad_norm'])
    D_vals = np.array(result['D'])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # --- Panel (a): 損失 ---
    ax = axes[0]
    ax.plot(losses, color=COLORS['nullspace'], linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss $J$')
    ax.set_title('(a) Loss trajectory')
    ax.grid(True, alpha=0.3)
    # 標註前期震盪
    ax.annotate('Initial oscillation',
                xy=(np.argmax(np.diff(losses) > 0) + 1, losses[len(losses)//4]),
                xytext=(len(losses)//3, losses[5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=9, color='red')

    # --- Panel (b): 梯度範數 ---
    ax = axes[1]
    ax.semilogy(grad_norms, color=COLORS['hungarian'], linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\|\nabla_a J\|$')
    ax.set_title('(b) Gradient norm decay')
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel (c): 失配度 D ---
    ax = axes[2]
    ax.plot(D_vals, color=COLORS['greedy'], linewidth=1.5)
    ax.axhline(0.02, color='red', linestyle='--', linewidth=1.5,
               label=r'$D_{th} = 0.02$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mismatch $D$')
    ax.set_title('(c) Mismatch evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path('figures/fig6_convergence.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    plt.close()


if __name__ == '__main__':
    main()