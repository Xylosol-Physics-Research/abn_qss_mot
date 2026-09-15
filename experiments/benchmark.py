# experiments/benchmark.py
"""
零空間投影 vs 匈牙利 vs 貪婪
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.constraint import MOTConstraintBuilder
from core.nullspace import NullSpaceExtractor
from core.solver import ProjectedGradientSolver
from core.binarize import binarize_hungarian, binarize_greedy


def benchmark(p: np.ndarray, tau: float = 0.2) -> dict:
    """對單一相似度矩陣進行三方法對比"""
    T, D = p.shape
    out = {}

    # 匈牙利
    t0 = time.perf_counter()
    row, col = linear_sum_assignment(-p)
    out['hungarian'] = {
        'time': time.perf_counter() - t0,
        'cost': float(-p[row, col].sum()),
        'n_matches': len(row),
    }

    # 貪婪
    t0 = time.perf_counter()
    bg = binarize_greedy(p, threshold=0.3)
    out['greedy'] = {
        'time': time.perf_counter() - t0,
        'cost': float(-(p * bg).sum()),
        'n_matches': int(bg.sum()),
    }

    # 零空間投影
    builder = MOTConstraintBuilder(T, D, p, tau=tau)
    M, _, _ = builder.build()
    extractor = NullSpaceExtractor(M)
    if extractor.K > 0:
        solver = ProjectedGradientSolver(
            M, extractor.null_basis, p, T, D,
            eta=0.01, gamma=0.2, D_th=0.02, max_iter=500
        )
        t0 = time.perf_counter()
        result = solver.solve()
        ns_time = time.perf_counter() - t0
        bn = binarize_hungarian(result['x'])
        out['nullspace'] = {
            'time': ns_time,
            'cost': float(-(p * bn).sum()),
            'n_matches': int(bn.sum()),
            'steps': result['step'],
            'K': extractor.K,
            'residual': result['residual'][-1],
        }
    else:
        out['nullspace'] = None

    return out


def run_sweep(sizes=[10, 20, 30, 50], n_trials=20):
    """掃描問題規模"""
    print(f"{'T,D':<8} {'方法':<12} {'時間(ms)':<14} {'成本':<12} {'步數':<8}")
    print("-" * 60)
    for size in sizes:
        T, D = size, size
        results = {'hungarian': [], 'greedy': [], 'nullspace': []}
        for _ in range(n_trials):
            p = np.random.rand(T, D)
            r = benchmark(p)
            for m in results:
                if r[m] is not None:
                    results[m].append(r[m])
        for m in ['hungarian', 'greedy', 'nullspace']:
            if not results[m]:
                continue
            times = [x['time'] * 1000 for x in results[m]]
            costs = [x['cost'] for x in results[m]]
            steps = [x.get('steps', 0) for x in results[m]]
            print(f"{size:<8} {m:<12} "
                  f"{np.mean(times):>6.2f}±{np.std(times):<5.2f} "
                  f"{np.mean(costs):>8.4f}  "
                  f"{np.mean(steps):>5.0f}")


if __name__ == '__main__':
    run_sweep(sizes=[10, 20, 30, 50], n_trials=20)
