import numpy as np
from scipy.linalg import svd

class NullSpaceExtractor:
    """零空間基底提取器"""

    def __init__(self, M: np.ndarray, tol: float = 1e-10):
        self.M = M
        self.tol = tol
        self.U, self.S, self.Vh = svd(M, full_matrices=True)
        self.rank = np.sum(self.S > tol)
        self.K = M.shape[1] - self.rank  # 零空間維度
        self.V = self.Vh.conj().T
        self.null_basis = self.V[:, -self.K:]  # (n+1) x K

    def project(self, C: np.ndarray) -> np.ndarray:
        """將任意向量投影到零空間"""
        P = self.null_basis @ self.null_basis.T
        return P @ C

    def check_constraint(self, C: np.ndarray) -> float:
        """檢查約束殘差 ||M C||"""
        return float(np.linalg.norm(self.M @ C))

    def verify_linearity(self, C_base: np.ndarray, delta: float = 0.01,
                          n_trials: int = 100) -> float:
        """
        驗證投影線性度 R^2（對應 ABN-QSS v3.0 的核心宣稱）
        在節點 0 上施加擾動 δ，測量響應線性度
        """
        responses = []
        perturbations = np.linspace(-delta, delta, n_trials)
        for eps in perturbations:
            C_pert = C_base.copy()
            C_pert[0] += eps
            C_final = self.project(C_pert)
            responses.append(np.linalg.norm(C_final - self.project(C_base)))
        responses = np.array(responses)
        # 線性擬合 R^2
        A = np.vstack([perturbations, np.ones_like(perturbations)]).T
        coeff, residuals, _, _ = np.linalg.lstsq(A, responses, rcond=None)
        y_pred = A @ coeff
        ss_res = np.sum((responses - y_pred) ** 2)
        ss_tot = np.sum((responses - responses.mean()) ** 2)
        R2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return R2, coeff[0]