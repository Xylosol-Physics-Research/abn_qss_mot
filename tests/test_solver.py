# tests/test_solver.py
import pytest
import numpy as np
from core.constraint import MOTConstraintBuilder
from core.nullspace import NullSpaceExtractor
from core.solver import ProjectedGradientSolver

class TestProjectedGradientSolver:

    def test_convergence_small(self):
        """測試小規模收斂"""
        T, D = 4, 4
        p = np.random.rand(T, D)
        builder = MOTConstraintBuilder(T, D, p, tau=0.1)
        M, _, _ = builder.build()
        extractor = NullSpaceExtractor(M)
        solver = ProjectedGradientSolver(
            M, extractor.null_basis, p, T, D,
            eta=0.01, gamma=0.2, D_th=0.02, max_iter=500
        )
        result = solver.solve()
        assert result['residual'][-1] < 1e-5
        assert result['D'][-1] < 0.02
        assert result['step'] < 500

    def test_monotonic_loss(self):
        """測試損失函數單調下降（李雅普諾夫穩定性）"""
        T, D = 5, 5
        p = np.random.rand(T, D)
        builder = MOTConstraintBuilder(T, D, p, tau=0.1)
        M, _, _ = builder.build()
        extractor = NullSpaceExtractor(M)
        solver = ProjectedGradientSolver(
            M, extractor.null_basis, p, T, D,
            eta=0.005, gamma=0.2, D_th=0.02, max_iter=200
        )
        result = solver.solve()
        losses = np.array(result['loss'])

        # 修正 1：確認損失確實下降
        assert losses[-1] < losses[0], "損失未下降"

        # 修正 2：檢查逐點單調性（允許浮點容差）
        diffs = np.diff(losses)
        assert np.all(diffs <= 1e-6), f"損失出現上升：{diffs[diffs > 1e-6]}"

        # 修正 3：僅在序列足夠長時比較前後半段
        if len(losses) >= 20:
            mid = len(losses) // 2
            assert np.mean(losses[mid:]) < np.mean(losses[:mid])

    def test_nonnegative_projection(self):
        """測試非負投影"""
        T, D = 3, 3
        p = np.random.rand(T, D)
        builder = MOTConstraintBuilder(T, D, p, tau=0.1)
        M, _, _ = builder.build()
        extractor = NullSpaceExtractor(M)
        solver = ProjectedGradientSolver(
            M, extractor.null_basis, p, T, D,
            eta=0.01, gamma=0.2, D_th=0.02, max_iter=100
        )
        result = solver.solve()
        x = result['x']
        assert np.all(x >= -1e-10), "x 存在負值"
        assert np.all(result['s'] >= -1e-10), "s 存在負值"
        assert np.all(result['t'] >= -1e-10), "t 存在負值"