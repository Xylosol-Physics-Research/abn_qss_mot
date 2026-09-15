# experiments/plotting/fig2_single_frame.py
"""
Figure 2: 單幀 MOT 基準
(a) 時間對比（log scale）
(b) 成本對比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from experiments.plotting.style import plt, COLORS, MARKERS

# 從實驗結果手動填入（或改為自動讀取）
SIZES = [10, 20, 30, 50]
DATA = {
    'hungarian': {'time': [0.01, 0.01, 0.03, 0.07],
                  'cost': [-8.60, -18.46, -28.51, -48.39]},
    'greedy':    {'time': [0.05, 0.16, 0.45, 1.57],
                  'cost': [-7.95, -17.52, -27.07, -46.77]},
    'nullspace': {'time': [34.56, 65.14, 222.27, 1498.87],
                  'cost': [-8.59, -18.46, -28.50, -48.37]},
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # --- Panel (a): 時間 ---
    ax = axes[0]
    for method in ['hungarian', 'greedy', 'nullspace']:
        ax.plot(SIZES, DATA[method]['time'],
                marker=MARKERS[method],
                color=COLORS[method],
                label=method.capitalize())
    ax.set_xlabel('Problem size $(T=D)$')
    ax.set_ylabel('Time (ms)')
    ax.set_yscale('log')
    ax.set_title('(a) Computation time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel (b): 成本 ---
    ax = axes[1]
    for method in ['hungarian', 'greedy', 'nullspace']:
        ax.plot(SIZES, DATA[method]['cost'],
                marker=MARKERS[method],
                color=COLORS[method],
                label=method.capitalize())
    ax.set_xlabel('Problem size $(T=D)$')
    ax.set_ylabel('Assignment cost')
    ax.set_title('(b) Solution quality')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path('figures/fig2_single_frame.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    plt.close()


if __name__ == '__main__':
    main()