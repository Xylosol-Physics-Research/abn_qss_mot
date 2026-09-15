# data/dataset.py
import os
import json
import numpy as np
from torch.utils.data import Dataset

class RESOLVEDataset(Dataset):
    """RESOLVE 協同感知數據集"""

    def __init__(self, root: str, split: str = 'train',
                 max_frames: int = None, min_tracks: int = 1):
        self.root = os.path.join(root, split)
        self.split = split
        self.max_frames = max_frames
        self.min_tracks = min_tracks
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        """加載所有序列元數據"""
        seqs = []
        for seq_name in sorted(os.listdir(self.root)):
            seq_path = os.path.join(self.root, seq_name)
            if not os.path.isdir(seq_path):
                continue
            meta_path = os.path.join(seq_path, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                seqs.append({'name': seq_name, 'path': seq_path, 'meta': meta})
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        frames = self._load_frames(seq['path'])
        return {'name': seq['name'], 'frames': frames, 'meta': seq['meta']}

    def _load_frames(self, seq_path: str):
        """加載單個序列的所有幀"""
        frames = []
        frame_files = sorted([f for f in os.listdir(seq_path) if f.startswith('frame_')])
        if self.max_frames:
            frame_files = frame_files[:self.max_frames]
        for ff in frame_files:
            frame_path = os.path.join(seq_path, ff)
            with open(frame_path) as f:
                frame = json.load(f)
            if len(frame.get('tracks', [])) >= self.min_tracks:
                frames.append(frame)
        return frames