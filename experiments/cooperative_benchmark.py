# experiments/cooperative_benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.cooperative import CooperativeSolver, binarize_x


def generate_data(M=3, T=8, D=10, overlap=0.4, seed=42):
    """生成協同感知數據"""
    rng = np.random.RandomState(seed)
    true_positions = rng.randn(T, 2) * 10
    vehicle_centers = rng.randn(M, 2) * 8
    
    # 每車視野
    observations = []
    true_ids = []
    for m in range(M):
        distances = np.linalg.norm(true_positions - vehicle_centers[m], axis=1)
        visible = distances < 14
        
        obs, tids = [], []
        for i in range(T):
            if visible[i]:
                obs.append(true_positions[i] + rng.randn(2) * 0.5)
                tids.append(i)
        while len(obs) < D:
            obs.append(rng.randn(2) * 10)
            tids.append(-1)
        observations.append(np.array(obs[:D]))
        true_ids.append(np.array(tids[:D]))
    
    # 相似度矩陣：本地軌跡（用真實位置） vs 本地檢測
    sim_locals = []
    for m in range(M):
        sim = np.zeros((T, D))
        for i in range(T):
            for j in range(D):
                d = np.linalg.norm(true_positions[i] - observations[m][j])
                sim[i, j] = np.exp(-d**2 / 2)
        sim_locals.append(sim)
    
    # 邊界檢測配對：同一真實目標在不同車的檢測
    boundary_pairs = []
    for m1 in range(M):
        for m2 in range(m1+1, M):
            for j1 in range(D):
                if true_ids[m1][j1] < 0:
                    continue
                for j2 in range(D):
                    if true_ids[m1][j1] == true_ids[m2][j2]:
                        boundary_pairs.append((m1, j1, m2, j2))
    
    return {
        'M': M, 'T': T, 'D': D,
        'true_positions': true_positions,
        'observations': observations,
        'true_ids': true_ids,
        'sim_locals': sim_locals,
        'boundary_pairs': boundary_pairs,
    }


def centralized_hungarian(data):
    """集中式匈牙利：理想基準"""
    M, T, D = data['M'], data['T'], data['D']
    # 把所有檢測堆成一個大矩陣
    all_obs = np.vstack(data['observations'])  # (M*D, 2)
    # 相似度：T × (M*D)
    sim_all = np.zeros((T, M*D))
    for i in range(T):
        for k in range(M*D):
            d = np.linalg.norm(data['true_positions'][i] - all_obs[k])
            sim_all[i, k] = np.exp(-d**2 / 2)
    
    row, col = linear_sum_assignment(-sim_all)
    assign = {k: i for i, k in zip(row, col) if sim_all[i, k] > 0.3}
    
    # 計算準確率
    correct, total = 0, 0
    for k, i in assign.items():
        m, j = k // D, k % D
        if data['true_ids'][m][j] >= 0:
            total += 1
            if data['true_ids'][m][j] == i:
                correct += 1
    return correct / max(total, 1)


def distributed_hungarian(data):
    """分散式匈牙利：每車獨立求解，無跨車通訊"""
    M, T, D = data['M'], data['T'], data['D']
    correct, total = 0, 0
    for m in range(M):
        row, col = linear_sum_assignment(-data['sim_locals'][m])
        for i, j in zip(row, col):
            if data['sim_locals'][m][i, j] > 0.3:
                if data['true_ids'][m][j] >= 0:
                    total += 1
                    if data['true_ids'][m][j] == i:
                        correct += 1
    return correct / max(total, 1)


def distributed_nullspace(data, max_iter=100):
    """分散式零空間：本地 + 邊界通訊"""
    M, T, D = data['M'], data['T'], data['D']
    solver = CooperativeSolver(M, T, D, max_iter=max_iter)
    x_locals, info = solver.solve_cooperative(
        data['sim_locals'], data['boundary_pairs']
    )
    
    correct, total = 0, 0
    for m in range(M):
        binary = binarize_x(x_locals[m])
        for i in range(T):
            for j in range(D):
                if binary[i, j] > 0.5:
                    if data['true_ids'][m][j] >= 0:
                        total += 1
                        if data['true_ids'][m][j] == i:
                            correct += 1
    return correct / max(total, 1), info


if __name__ == '__main__':
    print("=" * 85)
    print("協同感知基準測試")
    print("=" * 85)
    
    configs = [
        {'M': 2, 'T': 6, 'D': 8},
        {'M': 3, 'T': 8, 'D': 10},
        {'M': 4, 'T': 10, 'D': 12},
    ]
    
    print(f"\n{'配置':<16} {'方法':<16} {'時間(ms)':<12} "
          f"{'準確率':<10} {'通訊(bytes)':<12}")
    print("-" * 85)
    
    for cfg in configs:
        data = generate_data(**cfg)
        M, T, D = data['M'], data['T'], data['D']
        cfg_str = f"M={M},T={T},D={D}"
        
        # 集中式匈牙利
        t0 = time.perf_counter()
        h_acc = centralized_hungarian(data)
        h_time = (time.perf_counter() - t0) * 1000
        print(f"{cfg_str:<16} {'集中式匈牙利':<16} {h_time:<12.2f} "
              f"{h_acc:<10.3f} {'O(M*N)':<12}")
        
        # 分散式匈牙利
        t0 = time.perf_counter()
        dh_acc = distributed_hungarian(data)
        dh_time = (time.perf_counter() - t0) * 1000
        print(f"{'':<16} {'分散式匈牙利':<16} {dh_time:<12.2f} "
              f"{dh_acc:<10.3f} {'0':<12}")
        
        # 分散式零空間
        t0 = time.perf_counter()
        ns_acc, ns_info = distributed_nullspace(data)
        ns_time = (time.perf_counter() - t0) * 1000
        print(f"{'':<16} {'分散式零空間':<16} {ns_time:<12.2f} "
              f"{ns_acc:<10.3f} {ns_info['comm_bytes']:<12}")
        print()