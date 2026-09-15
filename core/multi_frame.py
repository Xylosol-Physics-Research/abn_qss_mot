# core/multi_frame.py（完整修正版）
"""
多幀聯合 MOT 的零空間投影求解器（修正版）

關鍵修正：
1. 移除錯誤的硬跨幀約束（原為互斥，語義錯誤）
2. 改用軟連續性約束（加入目標函數）
3. 使用 numpy.linalg.lstsq 兼容新版 SciPy
"""
import numpy as np
from scipy.linalg import svd


class MultiFrameConstraintBuilder:
    """K 幀聯合 MOT 約束構建器（僅單幀約束）"""

    def __init__(self, K, T, D, tau=0.2):
        self.K = K
        self.T = T
        self.D = D
        self.tau = tau

    def build(self):
        """僅構造單幀約束：檢測唯一性 + 軌跡唯一性"""
        K, T, D = self.K, self.T, self.D
        n_frame = T * D + D + T
        n_vars = K * n_frame

        rows, rhs = [], []

        for t in range(K):
            offset = t * n_frame
            # 檢測唯一性：Σ_i x^(t)_{ij} + s^(t)_j = 1
            for j in range(D):
                row = np.zeros(n_vars)
                for i in range(T):
                    row[offset + i * D + j] = 1.0
                row[offset + T * D + j] = 1.0
                rows.append(row)
                rhs.append(1.0)
            # 軌跡唯一性：Σ_j x^(t)_{ij} + u^(t)_i = 1
            for i in range(T):
                row = np.zeros(n_vars)
                for j in range(D):
                    row[offset + i * D + j] = 1.0
                row[offset + T * D + D + i] = 1.0
                rows.append(row)
                rhs.append(1.0)

        A = np.array(rows)
        b = np.array(rhs)

        if A.size == 0:
            raise ValueError(
                f"約束矩陣為空！K={K}, T={T}, D={D}"
            )

        return A, b, n_vars


class MultiFrameSolver:
    """多幀零空間投影求解器（含軟連續性約束）"""

    def __init__(self, A, b, n_vars, K, T, D,
                 eta=0.005, gamma=0.2, D_th=0.02, sigma_max2=10.0,
                 lambda_reg=0.01, mu_reg=0.1,
                 continuity_weight=0.5,
                 max_iter=300, min_iter=20, tol_loss=1e-7):
        self.A, self.b = A, b
        self.n_vars = n_vars
        self.K, self.T, self.D = K, T, D
        self.eta, self.gamma = eta, gamma
        self.D_th, self.sigma_max2 = D_th, sigma_max2
        self.lambda_reg, self.mu_reg = lambda_reg, mu_reg
        self.continuity_weight = continuity_weight
        self.max_iter, self.min_iter = max_iter, min_iter
        self.tol_loss = tol_loss

        # 特解 + 零空間基底（使用 numpy.linalg.lstsq 兼容新版）
        self.y0 = np.linalg.lstsq(A, b, rcond=None)[0]
        U, S, Vh = svd(A, full_matrices=True)
        rank = np.sum(S > 1e-10)
        self.V = Vh.conj().T[:, rank:]
        self.K_null = n_vars - rank

        if self.K_null == 0:
            raise ValueError(
                f"零空間為空！秩={rank}, 變量數={n_vars}。"
                f"請檢查約束數量。"
            )

    def _extract_xs(self, y):
        T, D, K = self.T, self.D, self.K
        n_frame = T * D + D + T
        xs = []
        for t in range(K):
            offset = t * n_frame
            xs.append(y[offset:offset + T * D].reshape(T, D))
        return xs

    def _objective(self, y, sims):
        xs = self._extract_xs(y)
        J = 0.0
        for t in range(self.K):
            J += -np.sum(sims[t] * xs[t])
            J += self.lambda_reg * np.sum(xs[t] ** 2)

        # 軟連續性約束：軌跡的 row-sum 應跨幀平滑
        if self.K > 1:
            row_sums = np.array([x.sum(axis=1) for x in xs])  # (K, T)
            diff = np.diff(row_sums, axis=0)                   # (K-1, T)
            J += self.continuity_weight * np.sum(diff ** 2)
        return J

    def _gradient_y(self, y, sims):
        T, D, K = self.T, self.D, self.K
        n_frame = T * D + D + T
        grad = np.zeros(self.n_vars)

        xs = self._extract_xs(y)
        for t in range(K):
            offset = t * n_frame
            x = xs[t]
            gx = -sims[t] + 2 * self.lambda_reg * x

            # 軟連續性對 x^(t) 的梯度
            if self.K > 1:
                row_sums = np.array([xx.sum(axis=1) for xx in xs])
                if t == 0:
                    gx += 2 * self.continuity_weight * (row_sums[0] - row_sums[1])[:, None]
                elif t == K - 1:
                    gx += 2 * self.continuity_weight * (row_sums[K-1] - row_sums[K-2])[:, None]
                else:
                    gx += 2 * self.continuity_weight * (
                        2 * row_sums[t] - row_sums[t-1] - row_sums[t+1]
                    )[:, None]

            grad[offset:offset + T * D] = gx.flatten()

            # 鬆弛變量梯度
            s = y[offset + T * D:offset + T * D + D]
            u = y[offset + T * D + D:offset + T * D + D + T]
            grad[offset + T * D:offset + T * D + D] = 2 * self.mu_reg * s
            grad[offset + T * D + D:offset + T * D + D + T] = 2 * self.mu_reg * u

        return grad

    def _mismatch(self, y):
        xs = self._extract_xs(y)
        vs = [np.var(x) for x in xs]
        return float(np.mean(vs) / self.sigma_max2)

    def solve(self, sims, verbose=False):
        z = np.zeros(self.K_null)
        y = self.y0 + self.V @ z
        loss_prev = self._objective(y, sims)
        history = {'loss': [loss_prev], 'step': 0,
                   'converged': False, 'y': None, 'x_matrices': None}

        for k in range(1, self.max_iter + 1):
            grad_y = self._gradient_y(y, sims)
            grad_z = self.V.T @ grad_y

            D_val = self._mismatch(y)
            a_mean = z.mean()
            a_norm = np.linalg.norm(z - a_mean) + 1e-8
            sym = self.gamma * (D_val - self.D_th) * (z - a_mean) / a_norm

            z = z - self.eta * grad_z - sym
            y = self.y0 + self.V @ z

            loss = self._objective(y, sims)
            history['loss'].append(loss)
            history['step'] = k

            if verbose and k % 50 == 0:
                print(f"[{k}] loss={loss:.6f}")

            if k >= self.min_iter and abs(loss_prev - loss) < self.tol_loss:
                history['converged'] = True
                break
            loss_prev = loss

        history['y'] = y
        history['x_matrices'] = self._extract_xs(y)
        return history