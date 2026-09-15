# eval/metrics.py
"""
MOT 標準指標：MOTA、MOTP、IDF1、ID Switch
"""
import numpy as np
from collections import defaultdict


def iou_3d(a, b):
    """3D 邊界框 IoU（簡化為中心距離 + 尺寸相似度）"""
    ca, sa = a[:3], a[3:6]
    cb, sb = b[:3], b[3:6]
    d = np.linalg.norm(ca - cb)
    size_sim = np.exp(-np.abs(sa - sb).sum() / 10.0)
    return size_sim * np.exp(-d ** 2 / 2.0)


def associate(pred_tracks, gt_tracks, iou_thresh=0.5):
    """匈牙利匹配預測與 GT"""
    from scipy.optimize import linear_sum_assignment
    T, G = len(pred_tracks), len(gt_tracks)
    if T == 0 or G == 0:
        return [], list(range(T)), list(range(G))

    iou_matrix = np.zeros((T, G))
    for i, pt in enumerate(pred_tracks):
        for j, gt in enumerate(gt_tracks):
            iou_matrix[i, j] = iou_3d(pt['box'], gt['box'])

    row, col = linear_sum_assignment(-iou_matrix)
    matches, unmatched_t, unmatched_g = [], [], []

    matched_t, matched_g = set(), set()
    for i, j in zip(row, col):
        if iou_matrix[i, j] >= iou_thresh:
            matches.append((i, j))
            matched_t.add(i)
            matched_g.add(j)

    unmatched_t = [i for i in range(T) if i not in matched_t]
    unmatched_g = [j for j in range(G) if j not in matched_g]
    return matches, unmatched_t, unmatched_g


def compute_mota(pred_frames, gt_frames):
    """MOTA = 1 - (FN + FP + IDSW) / GT"""
    total_gt = 0
    total_fn = 0
    total_fp = 0
    total_idsw = 0
    last_match = {}  # gt_id -> pred_id

    for pred, gt in zip(pred_frames, gt_frames):
        total_gt += len(gt)
        matches, unmatched_t, unmatched_g = associate(pred, gt)
        total_fn += len(unmatched_g)
        total_fp += len(unmatched_t)

        for i, j in matches:
            gt_id = gt[j]['id']
            pred_id = pred[i]['id']
            if gt_id in last_match and last_match[gt_id] != pred_id:
                total_idsw += 1
            last_match[gt_id] = pred_id

    if total_gt == 0:
        return 1.0
    mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt
    return max(-1.0, mota)


def count_id_switches(pred_frames, gt_frames):
    last_match = {}
    idsw = 0
    for pred, gt in zip(pred_frames, gt_frames):
        matches, _, _ = associate(pred, gt)
        for i, j in matches:
            gt_id = gt[j]['id']
            pred_id = pred[i]['id']
            if gt_id in last_match and last_match[gt_id] != pred_id:
                idsw += 1
            last_match[gt_id] = pred_id
    return idsw


def compute_idf1(pred_frames, gt_frames):
    """IDF1：簡化版本"""
    total_gt = sum(len(g) for g in gt_frames)
    if total_gt == 0:
        return 0.0
    correct = 0
    last_match = {}
    for pred, gt in zip(pred_frames, gt_frames):
        matches, _, _ = associate(pred, gt)
        for i, j in matches:
            gt_id = gt[j]['id']
            pred_id = pred[i]['id']
            if gt_id in last_match and last_match[gt_id] == pred_id:
                correct += 1
            last_match[gt_id] = pred_id
    precision = correct / max(sum(len(p) for p in pred_frames), 1)
    recall = correct / total_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)