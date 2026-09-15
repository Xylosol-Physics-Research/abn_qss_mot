# experiments/noise_robust.py
"""
對應 ABN-QSS v3.0 表2：噪聲強度 σ 對應同步適應度
"""
import numpy as np
from core.constraint import MOTConstraintBuilder
from core.nullspace import NullSpaceExtractor
from core.solver import ProjectedGradientSolver


def run_single(p_clean: np.ndarray, sigma: float, tau: float = 0.2) -> dict:
    T, D = p_clean.shape
    noise = np.random.normal(0, sigma, p_clean.shape)
    p = np.clip(p_clean + noise, 0, 1)

    builder = MOTConstraintBuilder(T, D, p, tau=tau)
    M, _, _ = builder.build()
    extractor = NullSpaceExtractor(M)

    if extractor.K == 0:
        return {'sigma': sigma, 'converged': False, 'D': 1.0, 'residual': 1.0}

    solver = ProjectedGradientSolver(
        M, extractor.null_basis, p, T, D,
        eta=0.01, gamma=0.2, D_th=0.02, max_iter=500
    )
    result = solver.solve()
    return {
        'sigma': sigma,
        'converged': result['residual'][-1] < 1e-6 and result['D'][-1] < 0.02,
        'D': result['D'][-1],
        'residual': result['residual'][-1],
        'steps': result['step'],
    }


def run_experiment(T=20, D=20, n_trials=20):
    sigmas = [0.00, 0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06]
    print(f"{'σ':<8} {'同步適應度':<12} {'D 終值':<12} {'殘差':<12} {'步數':<8}")
    print("-" * 60)
    for sigma in sigmas:
        results = []
        for _ in range(n_trials):
            p_clean = np.random.rand(T, D)
            results.append(run_single(p_clean, sigma))
        fitness = np.mean([1.0 if r['converged'] else 0.0 for r in results])
        D_final = np.mean([r['D'] for r in results])
        residual = np.mean([r['residual'] for r in results])
        steps = np.mean([r['steps'] for r in results])
        print(f"{sigma:<8.3f} {fitness:<12.3f} {D_final:<12.4f} "
              f"{residual:<12.2e} {steps:<8.0f}")


if __name__ == '__main__':
    run_experiment()