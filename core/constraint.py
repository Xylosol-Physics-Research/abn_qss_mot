import numpy as np
from typing import Tuple, List

class MOTConstraintBuilder:
    """MOT 約束矩陣 M 構造器"""

    def __init__(self, T: int, D: int, p: np.ndarray, tau: float = 0.2):
        """
        Args:
            T: 軌跡數
            D: 檢測數
            p: T x D 相似度矩陣，值域 [0, 1]
            tau: 硬約束閾值，低於此值禁止關聯
        """
        self.T = T
        self.D = D
        self.p = p
        self.tau = tau
        self.n = T * D + D + T  # 變量總數

    def build(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        返回:
            M: (m, n+1) 齊次約束矩陣
            b: (m,) 常數向量
            idx_hard: 硬約束索引列表
        """
        T, D, n = self.T, self.D, self.n

        # --- 約束 A: 檢測唯一性 sum_i x_ij + s_j = 1 ---
        A_det = np.zeros((D, n))
        for j in range(D):
            for i in range(T):
                A_det[j, i * D + j] = 1.0
            A_det[j, T * D + j] = 1.0  # s_j
        b_det = np.ones(D)

        # --- 約束 B: 軌跡唯一性 sum_j x_ij + t_i = 1 ---
        A_traj = np.zeros((T, n))
        for i in range(T):
            for j in range(D):
                A_traj[i, i * D + j] = 1.0
            A_traj[i, T * D + D + i] = 1.0  # t_i
        b_traj = np.ones(T)

        # --- 約束 C: 硬約束 x_ij = 0 ---
        idx_hard = [(i, j) for i in range(T) for j in range(D)
                    if self.p[i, j] < self.tau]
        A_hard = np.zeros((len(idx_hard), n))
        for k, (i, j) in enumerate(idx_hard):
            A_hard[k, i * D + j] = 1.0
        b_hard = np.zeros(len(idx_hard))

        # --- 合併 ---
        A = np.vstack([A_det, A_traj, A_hard]) if idx_hard else np.vstack([A_det, A_traj])
        b = np.concatenate([b_det, b_traj, b_hard]) if idx_hard else np.concatenate([b_det, b_traj])

        # --- 齊次化 M C = 0 ---
        M = np.hstack([A, -b.reshape(-1, 1)])
        return M, b, idx_hard

    def get_variable_slices(self) -> dict:
        """返回變量切片，便於後續提取 x, s, t"""
        T, D = self.T, self.D
        return {
            'x': slice(0, T * D),
            's': slice(T * D, T * D + D),
            't': slice(T * D + D, T * D + D + T),
        }