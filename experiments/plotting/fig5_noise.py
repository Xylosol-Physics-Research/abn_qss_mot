# experiments/plotting/fig5_noise.py
"""
Figure 5: 噪聲魯棒性
對應 ABN-QSS 表2：σ 對應有效算力
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from experiments.plotting.style import plt, COLORS

SIGMAS = np.array([0.00, 0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06])
# 從 ABN-QSS 表2 填入
FITNESS = np.array([1.000, 1.000, 1.000, 0.999, 0.800, 0.374, 0.133, 0.000])


def main():
    fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.plot(SIGMAS, FITNESS, marker='o', color=COLORS['nullspace'],
            linewidth=2, markersize=8, label='ZSP fitness')

    # 關鍵閾值
    ax.axvline(0.03, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(0.05, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(0.03, 0.5, 'Safety limit\n$\\sigma=0.03$',
            ha='right', va='center', fontsize=9, color='gray')
    ax.text(0.05, 0.3, 'Extreme\n$\\sigma=0.05$',
            ha='left', va='center', fontsize=9, color='red')

    # 關鍵區間填色
    ax.axvspan(0, 0.03, alpha=0.08, color='green', label='Safe zone')
    ax.axvspan(0.03, 0.05, alpha=0.08, color='orange',
               label='Graceful degradation')
    ax.axvspan(0.05, 0.06, alpha=0.08, color='red', label='Death zone')

    ax.set_xlabel(r'Noise intensity $\sigma$')
    ax.set_ylabel('Synchronization fitness')
    ax.set_title('Noise resilience of ZSP solver')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path('figures/fig5_noise.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    plt.close()


if __name__ == '__main__':
    main()