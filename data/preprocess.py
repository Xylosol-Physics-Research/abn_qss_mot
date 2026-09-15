# data/preprocess.py
"""
從 RESOLVE 原始數據提取：
- 3D 邊界框
- ReID 外觀嵌入
- 卡爾曼濾波運動預測
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class DetectionFeatureExtractor:
    """檢測特徵提取器"""

    def __init__(self, reid_model_path: str = None):
        self.reid_model = self._load_reid(reid_model_path) if reid_model_path else None

    def _load_reid(self, path):
        """加載 ReID 模型（如 ResNet50-IBN）"""
        model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        model.fc = torch.nn.Identity()  # 去掉分類頭
        model.eval()
        return model

    def extract_3d_box(self, det: dict) -> np.ndarray:
        """提取 3D 邊界框參數 [x, y, z, l, w, h, theta]"""
        return np.array([
            det['center_x'], det['center_y'], det['center_z'],
            det['length'], det['width'], det['height'],
            det['yaw'],
        ])

    def extract_appearance(self, image_patch: np.ndarray) -> np.ndarray:
        """提取 ReID 嵌入（128 維）"""
        if self.reid_model is None:
            return np.zeros(128)
        with torch.no_grad():
            tensor = torch.from_numpy(image_patch).float().permute(2, 0, 1).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(tensor, size=(256, 128))
            feat = self.reid_model(tensor).squeeze().numpy()
        return feat / (np.linalg.norm(feat) + 1e-8)

    def extract_motion(self, track_history: list) -> np.ndarray:
        """卡爾曼濾波預測下一幀位置"""
        # 簡化版：線性外推
        if len(track_history) < 2:
            return track_history[-1]['center']
        p_prev = track_history[-2]['center']
        p_curr = track_history[-1]['center']
        v = p_curr - p_prev
        return p_curr + v


class SimilarityMatrixBuilder:
    """相似度矩陣構建器"""

    def __init__(self, w_iou: float = 0.4, w_app: float = 0.3,
                 w_motion: float = 0.3, sigma: float = 1.0):
        self.w_iou = w_iou
        self.w_app = w_app
        self.w_motion = w_motion
        self.sigma = sigma

    def iou_3d(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """3D IoU（簡化為中心距離與尺寸比）"""
        center_a, size_a = box_a[:3], box_a[3:6]
        center_b, size_b = box_b[:3], box_b[3:6]
        dist = np.linalg.norm(center_a - center_b)
        size_sim = np.exp(-np.abs(size_a - size_b).sum() / 10.0)
        return size_sim * np.exp(-dist ** 2 / (2 * self.sigma ** 2))

    def appearance_sim(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """餘弦相似度"""
        return float(np.dot(feat_a, feat_b) / (np.linalg.norm(feat_a) * np.linalg.norm(feat_b) + 1e-8))

    def motion_sim(self, pred_a: np.ndarray, pos_b: np.ndarray) -> float:
        """運動預測相似度"""
        dist = np.linalg.norm(pred_a - pos_b)
        return float(np.exp(-dist ** 2 / (2 * self.sigma ** 2)))

    def build(self, tracks: list, detections: list) -> np.ndarray:
        """構建 T x D 相似度矩陣"""
        T, D = len(tracks), len(detections)
        p = np.zeros((T, D))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                p[i, j] = (self.w_iou * self.iou_3d(track['box'], det['box'])
                           + self.w_app * self.appearance_sim(track['feat'], det['feat'])
                           + self.w_motion * self.motion_sim(track['pred'], det['center']))
        return p