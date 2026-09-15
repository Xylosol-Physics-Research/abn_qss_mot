# experiments/plotting/plot_all.py
"""
一鍵繪製論文所有圖表
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from experiments.plotting import (
    fig1_digital_twin,
    fig2_single_frame,
    fig3_multiframe,
    fig4_cooperative,
    fig5_noise,
    fig6_convergence,
    fig7_scaling,
    fig8_boundary,
)


def main():
    figures = [
        ('Figure 1', fig1_digital_twin.main),
        ('Figure 2', fig2_single_frame.main),
        ('Figure 3', fig3_multiframe.main),
        ('Figure 4', fig4_cooperative.main),
        ('Figure 5', fig5_noise.main),
        ('Figure 6', fig6_convergence.main),
        ('Figure 7', fig7_scaling.main),
        ('Figure 8', fig8_boundary.main),
    ]
    for name, fn in figures:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            plt.close('all')


if __name__ == '__main__':
    main()