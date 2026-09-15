# experiments/plotting/fig8_boundary.py
"""
Figure 8: ZSP 適用邊界概念圖
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from experiments.plotting.style import plt


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # 中央分界線
    ax.plot([5, 5], [0.3, 6.7], 'k--', linewidth=2, alpha=0.5)

    # 左側：匈牙利擅長
    ax.text(2.5, 6.3, 'Linear Assignment Problems',
            ha='center', fontsize=12, fontweight='bold',
            color='#1f77b4')
    ax.text(2.5, 5.9, '(Hungarian dominates)',
            ha='center', fontsize=10, style='italic', color='#1f77b4')

    left_items = [
        'MOT data association',
        'Cooperative perception',
        'Bipartite matching',
        'Resource allocation',
    ]
    for i, item in enumerate(left_items):
        box = FancyBboxPatch((0.5, 4.7 - i*0.9), 4.0, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor='#1f77b4', alpha=0.15,
                             edgecolor='#1f77b4', linewidth=1.5)
        ax.add_patch(box)
        ax.text(2.5, 5.0 - i*0.9, item, ha='center', va='center',
                fontsize=10)

    # 右側：ZSP 擅長
    ax.text(7.5, 6.3, 'Nonlinear Constraint Satisfaction',
            ha='center', fontsize=12, fontweight='bold',
            color='#2ca02c')
    ax.text(7.5, 5.9, '(ZSP / ABN-QSS dominates)',
            ha='center', fontsize=10, style='italic', color='#2ca02c')

    right_items = [
        'Material simulation',
        'Quantum many-body',
        'Physics-layer crypto',
        'Molecular sensing',
    ]
    for i, item in enumerate(right_items):
        box = FancyBboxPatch((5.5, 4.7 - i*0.9), 4.0, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor='#2ca02c', alpha=0.15,
                             edgecolor='#2ca02c', linewidth=1.5)
        ax.add_patch(box)
        ax.text(7.5, 5.0 - i*0.9, item, ha='center', va='center',
                fontsize=10)

    # 底部註解
    ax.text(5, 0.4,
            'Key criterion: does the problem have a dual structure\n'
            'that enables an exact polynomial algorithm?',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 箭頭
    arrow = FancyArrowPatch((4.6, 5.0), (5.4, 5.0),
                            arrowstyle='<->', mutation_scale=20,
                            linewidth=2, color='gray')
    ax.add_patch(arrow)
    ax.text(5, 5.3, 'boundary', ha='center', fontsize=9,
            color='gray', fontweight='bold')

    plt.tight_layout()
    out = Path('figures/fig8_boundary.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    plt.close()


if __name__ == '__main__':
    main()