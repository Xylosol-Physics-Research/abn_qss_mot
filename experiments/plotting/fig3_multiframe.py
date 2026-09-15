# experiments/plotting/fig3_multiframe.py
"""
Figure 3: 多幀 MOT 統計檢驗
(a) 配對散點圖（匈牙利 vs ZSP）
(b) 差異分布直方圖
(c) 累積分布函數
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from experiments.plotting.style import plt, COLORS


def generate_multiframe_results(n_seeds=30, K=3, T=5, D=8, seed_base=0):
    """從 benchmark_multiframe 重新生成結果（或讀取緩存）"""
    from experiments.benchmark_multiframe import (
        generate_challenging_data, run_hungarian, run_multiframe, accuracy
    )
    h_accs, mf_accs = [], []
    for s in range(n_seeds):
        data = generate_challenging_data(
            K=K, T=T, D=D, seed=seed_base + s,
            miss_rate=0.25, false_rate=0.15, occlusion_prob=0.2
        )
        h_accs.append(accuracy(run_hungarian(data), data['det_true_ids']))
        mf_accs.append(accuracy(run_multiframe(data)[0], data['det_true_ids']))
    return np.array(h_accs), np.array(mf_accs)


def main():
    h_accs, mf_accs = generate_multiframe_results()
    diff = mf_accs - h_accs
    t_stat, p_val = stats.ttest_rel(mf_accs, h_accs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # --- Panel (a): 配對散點圖 ---
    ax = axes[0]
    ax.scatter(h_accs, mf_accs, alpha=0.6, s=40,
               color=COLORS['nullspace'], edgecolor='black', linewidth=0.5)
    lims = [min(h_accs.min(), mf_accs.min()) - 0.05,
            max(h_accs.max(), mf_accs.max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.2, label='$y=x$')
    ax.set_xlabel('Hungarian accuracy')
    ax.set_ylabel('ZSP accuracy')
    ax.set_title('(a) Paired accuracy')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Panel (b): 差異分布 ---
    ax = axes[1]
    ax.hist(diff, bins=15, color=COLORS['nullspace'],
            edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(diff.mean(), color='red', linestyle='-', linewidth=2,
               label=f'mean = {diff.mean():+.3f}')
    ax.set_xlabel('Accuracy difference (ZSP − Hungarian)')
    ax.set_ylabel('Count')
    ax.set_title(f'(b) Difference distribution\n'
                 f'p = {p_val:.4f}, winner = {(diff > 0).mean()*100:.0f}%')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Panel (c): 累積分布 ---
    ax = axes[2]
    sorted_diff = np.sort(diff)
    cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
    ax.plot(sorted_diff, cdf, marker='.', color=COLORS['nullspace'],
            linewidth=1.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Accuracy difference (ZSP − Hungarian)')
    ax.set_ylabel('CDF')
    ax.set_title('(c) Cumulative distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path('figures/fig3_multiframe.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.savefig(out.with_suffix('.png'))
    print(f"[OK] 已保存 {out}")
    print(f"    t={t_stat:.3f}, p={p_val:.4f}")
    plt.close()


if __name__ == '__main__':
    main()