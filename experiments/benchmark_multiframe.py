# experiments/benchmark_multiframe.py（完整修正版）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.multi_frame import MultiFrameConstraintBuilder, MultiFrameSolver


def generate_challenging_data(K=5, T=8, D=10, feat_dim=16,
                              seed=42, noise=0.5,
                              miss_rate=0.25,     # 25% 漏檢
                              false_rate=0.15,    # 15% 誤檢
                              occlusion_prob=0.2, # 20% 軌跡中斷
                              appearance_confusion=0.3):
    """
    生成有挑戰性的 MOT 數據：
    - 漏檢：真實軌跡在部分幀消失
    - 誤檢：憑空出現的檢測
    - 遮擋：軌跡連續性被破壞
    - 外觀混淆：相似外觀的干擾
    """
    rng = np.random.RandomState(seed)

    # 真實軌跡位置
    positions = np.zeros((T, K, 2))
    positions[:, 0, :] = rng.randn(T, 2) * 5
    velocities = rng.randn(T, 2) * 0.8
    for t in range(1, K):
        positions[:, t, :] = positions[:, t-1, :] + velocities

    # 真實軌跡外觀
    appearances = rng.randn(T, feat_dim)
    appearances /= np.linalg.norm(appearances, axis=1, keepdims=True)

    # 遮擋：每個軌跡有 occlusion_prob 機率在某幀消失
    # visible[i, t] = True 表示軌跡 i 在幀 t 可被檢測
    visible = rng.rand(T, K) > occlusion_prob
    visible[:, 0] = True  # 第一幀全部可見

    # 檢測生成
    det_positions = []
    det_appearances = []
    det_true_ids = []  # 每個檢測對應的真實軌跡 ID，-1 表示誤檢

    for t in range(K):
        frame_dets_pos = []
        frame_dets_app = []
        frame_true_ids = []

        # 真實軌跡對應的檢測（有漏檢）
        for i in range(T):
            if not visible[i, t]:
                continue
            if rng.rand() < miss_rate:
                continue  # 漏檢
            pos = positions[i, t] + rng.randn(2) * noise
            drift = rng.randn(feat_dim) * 0.15
            app = appearances[i] + drift
            app /= np.linalg.norm(app)
            frame_dets_pos.append(pos)
            frame_dets_app.append(app)
            frame_true_ids.append(i)

        # 誤檢（憑空出現）
        n_false = int(false_rate * D)
        for _ in range(n_false):
            # 位置隨機
            pos = rng.randn(2) * 8
            # 外觀可能是某真實軌跡的複製（混淆）
            if rng.rand() < appearance_confusion and T > 0:
                i_confuse = rng.randint(T)
                app = appearances[i_confuse] + rng.randn(feat_dim) * 0.2
            else:
                app = rng.randn(feat_dim)
            app /= np.linalg.norm(app)
            frame_dets_pos.append(pos)
            frame_dets_app.append(app)
            frame_true_ids.append(-1)

        # 填充到 D 個（補零或裁剪）
        while len(frame_dets_pos) < D:
            frame_dets_pos.append(rng.randn(2) * 10)
            frame_dets_app.append(rng.randn(feat_dim))
            frame_true_ids.append(-1)
        frame_dets_pos = frame_dets_pos[:D]
        frame_dets_app = frame_dets_app[:D]
        frame_true_ids = frame_true_ids[:D]

        det_positions.append(frame_dets_pos)
        det_appearances.append(frame_dets_app)
        det_true_ids.append(frame_true_ids)

    det_positions = np.array(det_positions)         # (K, D, 2)
    det_appearances = np.array(det_appearances)     # (K, D, feat_dim)
    det_true_ids = np.array(det_true_ids)           # (K, D)

    # 相似度矩陣：真實軌跡 vs 檢測
    sims = []
    for t in range(K):
        sim = np.zeros((T, D))
        for i in range(T):
            for j in range(D):
                motion_sim = np.exp(
                    -np.linalg.norm(positions[i, t] - det_positions[t, j])**2 / 2)
                app_sim = float(np.dot(appearances[i], det_appearances[t, j]))
                sim[i, j] = 0.5 * motion_sim + 0.5 * app_sim
        sims.append(sim)

    return {
        'positions': positions, 'appearances': appearances,
        'det_positions': det_positions, 'det_appearances': det_appearances,
        'det_true_ids': det_true_ids, 'sims': sims,
        'visible': visible,
        'K': K, 'T': T, 'D': D,
    }


def run_hungarian(data):
    """逐幀匈牙利"""
    K, D = data['K'], data['D']
    assignments = []
    for t in range(K):
        row, col = linear_sum_assignment(-data['sims'][t])
        assign = np.full(D, -1)
        for i, j in zip(row, col):
            if data['sims'][t][i, j] > 0.3:  # 相似度門檻
                assign[j] = i
        assignments.append(assign)
    return assignments


def run_multiframe(data, verbose=False):
    """多幀零空間投影"""
    K, T, D = data['K'], data['T'], data['D']
    builder = MultiFrameConstraintBuilder(K, T, D, tau=0.2)
    A, b, n_vars = builder.build()

    if verbose:
        print(f"  約束數: {A.shape[0]}, 變量數: {n_vars}")

    solver = MultiFrameSolver(
        A, b, n_vars, K, T, D,
        eta=0.005, gamma=0.2, D_th=0.02,
        continuity_weight=0.5,
        max_iter=300, min_iter=20
    )

    result = solver.solve(data['sims'], verbose=verbose)

    assignments = []
    for x in result['x_matrices']:
        row, col = linear_sum_assignment(-x)
        assign = np.full(D, -1)
        for i, j in zip(row, col):
            if x[i, j] > 0.3:
                assign[j] = i
        assignments.append(assign)
    return assignments, result


def count_switches(assignments, det_true_ids):
    """計算 ID 切換次數（只計算真實軌跡的切換）"""
    K, D = det_true_ids.shape
    true_to_pred = {}
    for t in range(K):
        for j in range(D):
            true_id = det_true_ids[t, j]
            if true_id < 0:
                continue
            pred_id = assignments[t][j]
            if pred_id < 0:
                continue
            true_to_pred.setdefault(true_id, []).append((t, pred_id))
    switches = 0
    for history in true_to_pred.values():
        history.sort()
        for k in range(1, len(history)):
            if history[k][1] != history[k-1][1]:
                switches += 1
    return switches


def accuracy(assignments, det_true_ids):
    """準確率：只計算真實軌跡對應的檢測"""
    K, D = det_true_ids.shape
    correct = total = 0
    for t in range(K):
        for j in range(D):
            if det_true_ids[t, j] >= 0:
                total += 1
                if assignments[t][j] == det_true_ids[t, j]:
                    correct += 1
    return correct / max(total, 1)


if __name__ == '__main__':
    print("=" * 85)
    print("多幀聯合 MOT 基準測試（挑戰性數據）")
    print("=" * 85)

    configs = [
        {'K': 3, 'T': 5, 'D': 8, 'seed': 42,
         'miss_rate': 0.25, 'false_rate': 0.15, 'occlusion_prob': 0.2},
        {'K': 5, 'T': 6, 'D': 10, 'seed': 43,
         'miss_rate': 0.3, 'false_rate': 0.2, 'occlusion_prob': 0.25},
        {'K': 4, 'T': 8, 'D': 12, 'seed': 44,
         'miss_rate': 0.3, 'false_rate': 0.2, 'occlusion_prob': 0.3},
    ]

    print(f"\n{'配置':<18} {'方法':<14} {'時間(ms)':<12} "
          f"{'準確率':<10} {'ID切換':<10}")
    print("-" * 85)

    for cfg in configs:
        data = generate_challenging_data(**cfg)
        K, T, D = data['K'], data['T'], data['D']

        t0 = time.perf_counter()
        h_assign = run_hungarian(data)
        h_time = (time.perf_counter() - t0) * 1000
        h_acc = accuracy(h_assign, data['det_true_ids'])
        h_sw = count_switches(h_assign, data['det_true_ids'])

        t0 = time.perf_counter()
        mf_assign, mf_result = run_multiframe(data)
        mf_time = (time.perf_counter() - t0) * 1000
        mf_acc = accuracy(mf_assign, data['det_true_ids'])
        mf_sw = count_switches(mf_assign, data['det_true_ids'])

        cfg_str = f"K={K},T={T},D={D}"
        print(f"{cfg_str:<18} {'匈牙利':<14} {h_time:<12.2f} "
              f"{h_acc:<10.3f} {h_sw:<10}")
        print(f"{'':<18} {'多幀零空間':<14} {mf_time:<12.2f} "
              f"{mf_acc:<10.3f} {mf_sw:<10}")
        print(f"{'':<18} 收斂={mf_result['converged']}, "
              f"步數={mf_result['step']}")
        print()