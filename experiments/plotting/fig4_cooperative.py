# experiments/plotting/fig4_cooperative.py
"""
Figure 4: 協同感知基準
(a) 時間對比（log scale）
(b) 通訊量對比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from experiments.plotting.style import plt, COLORS, MARKERS

# 從協同感知基準填入（每車 T×8 bytes 通訊）
CONFIGS = ['M=2\nT=6,D=8', 'M=3\nT=8,D=10', 'M=4\nT=10,D=12']
DATA = {
    'central':     {'time': [21.71, 0.82, 1.56],
                    'comm': [48, 80, 120]},  # M*T*8 bytes 抽象
    'distributed': {'time': [0.03, 0.04, 0.05],
                    'comm': [0, 0, 0]},
    'zsp':         {'time': [15.99, 26.22, 71.23],
                    'comm': [480, 960, 1600]},
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(CONFIGS))

    # --- Panel (a): 時間 ---
    ax = axes[0]
    for method, label in [('central', 'Centralized Hungarian'),
                          ('distributed', 'Distributed Hungarian'),
                          ('zsp', 'Distributed ZSP')]:
        ax.plot(x, DATA[method]['time'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS[method], label=label, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylabel('Time (ms)')
    ax.set_yscale('log')
    ax.set_title('(a) Computation time')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel (b): 通訊量 ---
    ax = axes[1]
    width = 0.26
    ax.bar(x - width, DATA['central']['comm'], width,
           label='Centralized', color=COLORS['central'], edgecolor='black')
    ax.bar(x, DATA['distributed']['comm'], width,
           label='Distributed', color=COLORS['distributed'], edgecolor='black')
    ax.bar(x + width, DATA['zsp']['comm'], width,
           label='ZSP', color=COLORS['zsp'], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylabel('Communication (bytes)')
    ax.set_title('(b) Communication overhead')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = Path('figures/fig4_cooperative.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    plt.close()


if __name__ == '__main__':
    main()