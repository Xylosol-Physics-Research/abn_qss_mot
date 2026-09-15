# core/cooperative.py
"""
協同感知零空間投影
"""
import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment


class CooperativeSolver:
    """
    協同感知分散式求解器
    
    核心思想：
    - 每車本地構建零空間約束
    - 僅交換邊界檢測的低維特徵
    - 迭代收斂至全域一致
    """
    
    def __init__(self, M, T, D, eta=0.02, max_iter=200,
                 boundary_thresh=2.0):
        self.M = M
        self.T = T
        self.D = D
        self.eta = eta
        self.max_iter = max_iter
        self.boundary_thresh = boundary_thresh
        self.comm_bytes = 0  # 通訊量累計
    
    def _build_local_constraint(self, T, D):
        """單車本地約束"""
        n_frame = T * D + D + T
        rows, rhs = [], []
        
        # 檢測唯一性
        for j in range(D):
            row = np.zeros(n_frame)
            for i in range(T):
                row[i * D + j] = 1.0
            row[T * D + j] = 1.0
            rows.append(row)
            rhs.append(1.0)
        
        # 軌跡唯一性
        for i in range(T):
            row = np.zeros(n_frame)
            for j in range(D):
                row[i * D + j] = 1.0
            row[T * D + D + i] = 1.0
            rows.append(row)
            rhs.append(1.0)
        
        return np.array(rows), np.array(rhs), n_frame
    
    def _extract_local_x(self, y, T, D):
        """從 y 提取 x"""
        n_frame = T * D + D + T
        return y[:T*D].reshape(T, D)
    
    def solve_vehicle(self, sim_local, neighbor_info=None):
        """
        單車求解
        
        neighbor_info: list of dict {
            'boundary_idx': 本地檢測索引,
            'neighbor_det': 鄰車檢測位置,
            'neighbor_feat': 鄰車檢測特徵
        }
        """
        T, D = self.T, self.D
        A, b, n_frame = self._build_local_constraint(T, D)
        
        # 特解 + 零空間
        y0 = np.linalg.lstsq(A, b, rcond=None)[0]
        U, S, Vh = svd(A, full_matrices=True)
        rank = np.sum(S > 1e-10)
        V = Vh.conj().T[:, rank:]
        K_null = n_frame - rank
        
        if K_null == 0:
            return None
        
        z = np.zeros(K_null)
        y = y0 + V @ z
        
        # 跨車軟約束（若有鄰車資訊）
        cross_constraint_weight = 0.3
        if neighbor_info:
            cross_bonus = np.zeros((T, D))
            for info in neighbor_info:
                d_local = info['boundary_idx']
                # 本地檢測 d_local 與鄰車的關聯啟發
                # 若鄰車檢測特徵與本地軌跡相似，加權
                for i in range(T):
                    sim_score = info.get('similarity', [0]*T)
                    cross_bonus[i, d_local] += sim_score[i]
            cross_bonus = cross_bonus / max(len(neighbor_info), 1)
        else:
            cross_bonus = np.zeros((T, D))
        
        for k in range(self.max_iter):
            x = self._extract_local_x(y, T, D)
            
            # 目標：最大化 Σ sim · x + 跨車 bonus
            grad_x = -(sim_local + cross_constraint_weight * cross_bonus)
            grad_x += 2 * 0.01 * x  # 正則化
            
            # 擴展到 y 空間
            grad_y = np.zeros(n_frame)
            grad_y[:T*D] = grad_x.flatten()
            # s 和 t 的梯度
            s = y[T*D:T*D+D]
            u = y[T*D+D:]
            grad_y[T*D:T*D+D] = 2 * 0.1 * s
            grad_y[T*D+D:] = 2 * 0.1 * u
            
            grad_z = V.T @ grad_y
            z = z - self.eta * grad_z
            y = y0 + V @ z
        
        return self._extract_local_x(y, T, D)
    
    def solve_cooperative(self, sim_locals, boundary_pairs):
        """
        協同求解
        
        boundary_pairs: list of (vehicle_a, idx_a, vehicle_b, idx_b)
                        表示兩車的邊界檢測配對
        """
        M = self.M
        
        # 初始：每車獨立求解
        x_locals = []
        for m in range(M):
            x = self.solve_vehicle(sim_locals[m])
            x_locals.append(x)
        
        # 迭代：邊界一致性
        for iteration in range(5):
            # 每車將邊界檢測的關聯結果廣播給鄰車
            boundary_broadcasts = []
            for m in range(M):
                broadcast = {}
                for (va, ia, vb, ib) in boundary_pairs:
                    if va == m:
                        broadcast[(vb, ib)] = x_locals[m][:, ia]
                    elif vb == m:
                        broadcast[(va, ia)] = x_locals[m][:, ib]
                boundary_broadcasts.append(broadcast)
                self.comm_bytes += self.T * 8  # 每車 T 個 float64
            
            # 每車根據鄰車資訊重新求解
            new_x_locals = []
            for m in range(M):
                neighbor_info = []
                for (va, ia, vb, ib) in boundary_pairs:
                    if va == m and (vb, ib) in boundary_broadcasts[m]:
                        neighbor_info.append({
                            'boundary_idx': ia,
                            'similarity': boundary_broadcasts[m][(vb, ib)],
                        })
                    elif vb == m and (va, ia) in boundary_broadcasts[m]:
                        neighbor_info.append({
                            'boundary_idx': ib,
                            'similarity': boundary_broadcasts[m][(va, ia)],
                        })
                new_x = self.solve_vehicle(sim_locals[m], neighbor_info)
                new_x_locals.append(new_x)
            
            # 檢查收斂
            delta = sum(
                np.linalg.norm(new_x_locals[m] - x_locals[m])
                for m in range(M)
            )
            x_locals = new_x_locals
            if delta < 1e-3:
                break
        
        return x_locals, {
            'iterations': iteration + 1,
            'comm_bytes': self.comm_bytes,
        }


def binarize_x(x):
    """匈牙利二值化"""
    T, D = x.shape
    row, col = linear_sum_assignment(-x)
    binary = np.zeros_like(x)
    for i, j in zip(row, col):
        if x[i, j] > 0.3:
            binary[i, j] = 1.0
    return binary