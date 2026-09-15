# core/solver.py
import numpy as np
from typing import Optional


class ProjectedGradientSolver:
    """
    神曲引擎數字孿生：投影梯度求解器（修正版）
    
    關鍵修正：
    1. 移除「殘差判據」（零空間參數化下恒為 0）
    2. 改用「目標函數變化量」作為收斂判據
    3. 強制最小迭代次數
    4. 記錄完整損失歷史供診斷
    """

    def __init__(self, M, null_basis, p, T, D,
                 eta=0.01, gamma=0.2, D_th=0.02, sigma_max2=10.0,
                 lambda_reg=0.01, mu_reg=0.1,
                 max_iter=300, min_iter=20,
                 tol_loss=1e-6, tol_grad=1e-8):
        self.M = M
        self.V = null_basis
        self.K = null_basis.shape[1]
        self.p = p
        self.T = T
        self.D = D
        self.eta = eta
        self.gamma = gamma
        self.D_th = D_th
        self.sigma_max2 = sigma_max2
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol_loss = tol_loss
        self.tol_grad = tol_grad

    def _extract_vars(self, C):
        T, D = self.T, self.D
        x = C[:T * D].reshape(T, D)
        s = C[T * D:T * D + D]
        t = C[T * D + D:T * D + D + T]
        return x, s, t

    def _objective(self, C):
        x, s, t = self._extract_vars(C)
        # 主要目標：最大化 Σ p_ij x_ij（即最小化負值）
        # 正則化：避免 x 過大
        return (-np.sum(self.p * x)
                + self.lambda_reg * np.sum(x ** 2)
                + self.mu_reg * (np.sum(s ** 2) + np.sum(t ** 2)))

    def _gradient(self, C):
        T, D = self.T, self.D
        x, s, t = self._extract_vars(C)
        grad = np.zeros_like(C)
        grad[:T * D] = (-self.p + 2 * self.lambda_reg * x).flatten()
        grad[T * D:T * D + D] = 2 * self.mu_reg * s
        grad[T * D + D:T * D + D + T] = 2 * self.mu_reg * t
        return grad

    def _mismatch(self, C):
        x, _, _ = self._extract_vars(C)
        return float(np.var(x) / self.sigma_max2)

    def _project_nonneg(self, C):
        """非負投影，對應 OTA 飽和鉗位"""
        T, D = self.T, self.D
        C = C.copy()
        C[:T * D] = np.maximum(C[:T * D], 0.0)
        C[T * D:T * D + D] = np.maximum(C[T * D:T * D + D], 0.0)
        C[T * D + D:T * D + D + T] = np.maximum(
            C[T * D + D:T * D + D + T], 0.0)
        return C

    def solve(self, a0=None, verbose=False):
        """執行投影梯度下降"""
        if a0 is None:
            a0 = np.ones(self.K) / self.K
        a = a0.copy()

        # 初始評估
        C = self.V @ a
        loss_prev = self._objective(C)
        history = {
            'loss': [loss_prev],
            'D': [self._mismatch(C)],
            'grad_norm': [],
            'step': 0,
            'converged': False,
            'reason': None,
        }

        for k in range(1, self.max_iter + 1):
            # --- 梯度計算（純 a 空間）---
            grad_C = self._gradient(C)
            grad_a = self.V.T @ grad_C
            grad_norm = float(np.linalg.norm(grad_a))

            # --- 神曲引擎更新 ---
            D_val = self._mismatch(C)
            a_mean = a.mean()
            a_norm = np.linalg.norm(a - a_mean) + 1e-8
            symmetry_term = (self.gamma * (D_val - self.D_th)
                             * (a - a_mean) / a_norm)
            a = a - self.eta * grad_a - symmetry_term

            # --- 重新評估 ---
            C = self.V @ a
            loss = self._objective(C)
            D_val = self._mismatch(C)

            history['loss'].append(loss)
            history['D'].append(D_val)
            history['grad_norm'].append(grad_norm)
            history['step'] = k

            if verbose and k % 20 == 0:
                print(f"[{k:4d}] loss={loss:.6f} Δloss={loss_prev - loss:.2e} "
                      f"D={D_val:.6f} ‖∇‖={grad_norm:.2e}")

            # --- 收斂判據（修正版）---
            loss_change = abs(loss_prev - loss)

            if k >= self.min_iter:
                if loss_change < self.tol_loss:
                    history['converged'] = True
                    history['reason'] = 'loss_change'
                    break
                if grad_norm < self.tol_grad:
                    history['converged'] = True
                    history['reason'] = 'grad_norm'
                    break

            loss_prev = loss

        # --- 最後一次非負投影（OTA 鉗位）---
        C = self.V @ a
        C = self._project_nonneg(C)
        # 投影回零空間
        a = np.linalg.lstsq(self.V, C, rcond=None)[0]
        C = self.V @ a

        history['final_C'] = C
        history['final_a'] = a
        x, s, t = self._extract_vars(C)
        history['x'] = x
        history['s'] = s
        history['t'] = t
        history['final_loss'] = self._objective(C)
        # history['residual'] = float(np.linalg.norm(self.M @ C))  # 供記錄
        history['residual'] = [float(np.linalg.norm(self.M @ C))]
        return history