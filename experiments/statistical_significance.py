# experiments/statistical_significance.py
import numpy as np
from experiments.benchmark_multiframe import (
    generate_challenging_data, run_hungarian, run_multiframe,
    accuracy, count_switches
)

def run_statistical(n_seeds=20, K=3, T=5, D=8):
    """多種子統計測試"""
    h_accs, mf_accs = [], []
    h_sws, mf_sws = [], []
    for seed in range(n_seeds):
        data = generate_challenging_data(
            K=K, T=T, D=D, seed=seed,
            miss_rate=0.25, false_rate=0.15, occlusion_prob=0.2
        )
        h_assign = run_hungarian(data)
        mf_assign, _ = run_multiframe(data)
        h_accs.append(accuracy(h_assign, data['det_true_ids']))
        mf_accs.append(accuracy(mf_assign, data['det_true_ids']))
        h_sws.append(count_switches(h_assign, data['det_true_ids']))
        mf_sws.append(count_switches(mf_assign, data['det_true_ids']))

    h_accs = np.array(h_accs)
    mf_accs = np.array(mf_accs)
    diff = mf_accs - h_accs

    # 配對 t 檢驗
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(mf_accs, h_accs)

    print(f"匈牙利:      {h_accs.mean():.3f} ± {h_accs.std():.3f}")
    print(f"多幀零空間:  {mf_accs.mean():.3f} ± {mf_accs.std():.3f}")
    print(f"平均差異:    {diff.mean():+.3f} ± {diff.std():.3f}")
    print(f"t 統計量:    {t_stat:.3f}")
    print(f"p 值:        {p_value:.4f}  {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"多幀勝出比例: {(diff > 0).mean()*100:.1f}%")

if __name__ == '__main__':
    run_statistical(n_seeds=30, K=3, T=5, D=8)