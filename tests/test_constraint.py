# tests/test_constraint.py
import pytest
import numpy as np
from core.constraint import MOTConstraintBuilder

class TestMOTConstraintBuilder:

    def test_basic_construction(self):
        """測試基本約束矩陣構造"""
        T, D = 3, 4
        p = np.random.rand(T, D)
        builder = MOTConstraintBuilder(T, D, p, tau=0.2)
        M, b, idx_hard = builder.build()
        # 檢查維度
        assert M.shape == (D + T + len(idx_hard), T * D + D + T + 1)
        assert b.shape == (D + T + len(idx_hard),)
        # 檢查約束 A: sum_i x_ij + s_j = 1
        for j in range(D):
            row = M[j, :T * D].reshape(T, D)
            assert np.allclose(row[:, j], 1.0)
            assert M[j, T * D + j] == 1.0
        # 檢查約束 B: sum_j x_ij + t_i = 1
        for i in range(T):
            row = M[D + i, :T * D].reshape(T, D)
            assert np.allclose(row[i, :], 1.0)
            assert M[D + i, T * D + D + i] == 1.0

    def test_hard_constraints(self):
        """測試硬約束正確生成"""
        T, D = 3, 4
        p = np.array([[0.1, 0.5, 0.3, 0.8],
                      [0.9, 0.2, 0.4, 0.1],
                      [0.3, 0.7, 0.2, 0.6]])
        builder = MOTConstraintBuilder(T, D, p, tau=0.2)
        M, b, idx_hard = builder.build()
        # p[0,0]=0.1<0.2, p[1,1]=0.2 (不小於), p[2,2]=0.2
        assert (0, 0) in idx_hard
        assert (1, 1) not in idx_hard  # 0.2 不小於 0.2

    def test_zero_space_nonempty(self):
        """測試零空間非空"""
        T, D = 5, 5
        p = np.random.rand(T, D)
        builder = MOTConstraintBuilder(T, D, p, tau=0.1)
        M, b, _ = builder.build()
        from scipy.linalg import svd
        _, S, _ = svd(M)
        rank = np.sum(S > 1e-10)
        K = M.shape[1] - rank
        assert K > 0, "零空間為空，約束衝突"

    def test_variable_slices(self):
        """測試變量切片正確性"""
        T, D = 3, 4
        builder = MOTConstraintBuilder(T, D, np.random.rand(T, D))
        slices = builder.get_variable_slices()
        assert slices['x'] == slice(0, 12)
        assert slices['s'] == slice(12, 16)
        assert slices['t'] == slice(16, 19)