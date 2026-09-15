# experiments/plotting/fig7_scaling.py
"""
Figure 7: 問題規模 vs 時間（log-log）
顯示匈牙利 O(n³) vs ZSP O(n²·iter) 的複雜度差異
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from experiments.plotting.style import plt, COLORS, MARKERS

SIZES = np.array([10, 20, 30, 50])
TIME_HUNGARIAN = np.array([0.01, 0.01, 0.03, 0.07])
TIME_ZSP = np.array([34.56, 65.14, 222.27, 1498.87])
TIME_GREEDY = np.array([0.05, 0.16, 0.45, 1.57])


def fit_power_law(x, y):
    """擬合 y = a * x^b"""
    log_x, log_y = np.log(x), np.log(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    return np.exp(log_a), b


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.loglog(SIZES, TIME_HUNGARIAN, marker=MARKERS['hungarian'],
              color=COLORS['hungarian'], linewidth=2,
              label='Hungarian')
    ax.loglog(SIZES, TIME_GREEDY, marker=MARKERS['greedy'],
              color=COLORS['greedy'], linewidth=2,
              label='Greedy')
    ax.loglog(SIZES, TIME_ZSP, marker=MARKERS['nullspace'],
              color=COLORS['nullspace'], linewidth=2,
              label='ZSP')

    # 擬合冪次
    a_h, b_h = fit_power_law(SIZES, TIME_HUNGARIAN)
    a_z, b_z = fit_power_law(SIZES, TIME_ZSP)
    ax.text(0.05, 0.95,
            f'Hungarian: $O(n^{{{b_h:.2f}}})$',
            transform=ax.transAxes, fontsize=9,
            color=COLORS['hungarian'], va='top')
    ax.text(0.05, 0.88,
            f'ZSP: $O(n^{{{b_z:.2f}}})$',
            transform=ax.transAxes, fontsize=9,
            color=COLORS['nullspace'], va='top')

    ax.set_xlabel('Problem size $(T=D)$')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Scaling behavior: ZSP vs classical methods')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out = Path('figures/fig7_scaling.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    print(f"    Hungarian: O(n^{b_h:.2f})")
    print(f"    ZSP:       O(n^{b_z:.2f})")
    plt.close()


if __name__ == '__main__':
    main()