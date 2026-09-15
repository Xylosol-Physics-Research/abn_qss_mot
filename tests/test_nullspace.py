# tests/test_nullspace.py
import pytest
import numpy as np
from core.nullspace import NullSpaceExtractor

class TestNullSpaceExtractor:

    def test_rank_and_dimension(self):
        """測試秩與零空間維度"""
        # 構造已知零空間的矩陣
        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        extractor = NullSpaceExtractor(M)
        assert extractor.rank == 2
        assert extractor.K == 2
        assert extractor.null_basis.shape == (4, 2)

    def test_projection_idempotent(self):
        """測試投影矩陣冪等性 P^2 = P"""
        M = np.random.rand(3, 8)
        extractor = NullSpaceExtractor(M)
        C = np.random.rand(8)
        P_C = extractor.project(C)
        P_P_C = extractor.project(P_C)
        np.testing.assert_allclose(P_C, P_P_C, atol=1e-10)

    def test_projection_orthogonal(self):
        """測試投影後向量在零空間"""
        M = np.random.rand(3, 8)
        extractor = NullSpaceExtractor(M)
        C = np.random.rand(8)
        P_C = extractor.project(C)
        residual = np.linalg.norm(M @ P_C)
        assert residual < 1e-10, f"投影殘差 {residual} 過大"

    def verify_linearity(self, C_base, delta=0.01, n_trials=200):
        """
        驗證投影線性度：沿 P·e₀ 方向的帶符號響應
        """
        P = self.null_basis @ self.null_basis.T
        e0 = np.zeros_like(C_base)
        e0[0] = 1.0
        direction = P @ e0
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-12:
            return 0.0, 0.0
        direction = direction / dir_norm  # 單位方向

        P_C_base = P @ C_base
        responses = []
        perturbations = np.linspace(-delta, delta, n_trials)
        for eps in perturbations:
            C_pert = C_base.copy()
            C_pert[0] += eps
            C_final = P @ C_pert
            # 關鍵修正：帶符號的標量響應，而非範數
            responses.append(np.dot(C_final - P_C_base, direction))

        responses = np.array(responses)
        A = np.vstack([perturbations, np.ones_like(perturbations)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, responses, rcond=None)
        y_pred = A @ coeff
        ss_res = np.sum((responses - y_pred) ** 2)
        ss_tot = np.sum((responses - responses.mean()) ** 2)
        R2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return R2, coeff[0]