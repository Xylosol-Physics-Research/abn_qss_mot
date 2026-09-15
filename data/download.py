# data/download.py
"""
多數據集下載器：RESOLVE / Griffin / TUMTraf
支援斷點續傳與校驗
"""
import os
import zipfile
import hashlib
import requests
from tqdm import tqdm
from pathlib import Path

DATASETS = {
    'resolve': {
        'urls': {
            'train': 'https://resolve-dataset.org/download/train.zip',
            'val':   'https://resolve-dataset.org/download/val.zip',
            'test':  'https://resolve-dataset.org/download/test.zip',
        },
        'md5': {
            'train': None,  # 申請後填寫
            'val':   None,
            'test':  None,
        },
    },
    'griffin': {
        'urls': {
            'all': 'https://github.com/griffin-dataset/griffin/releases/download/v1.0/griffin.zip',
        },
        'md5': {'all': None},
    },
    'tumtraf': {
        'urls': {
            'all': 'https://tum-traffic-dataset.github.io/download/tumtraf_v2x.zip',
        },
        'md5': {'all': None},
    },
}

def md5sum(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

def download_file(url: str, save_path: str, expected_md5: str = None):
    """斷點續傳下載 + MD5 校驗"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and expected_md5:
        if md5sum(save_path) == expected_md5:
            print(f"[Skip] 已存在且校驗通過：{save_path}")
            return save_path

    headers = {}
    mode = 'wb'
    pos = 0
    if save_path.exists():
        pos = save_path.stat().st_size
        headers['Range'] = f'bytes={pos}-'
        mode = 'ab'

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0)) + pos

    with open(save_path, mode) as f, tqdm(
        desc=save_path.name, total=total, initial=pos,
        unit='iB', unit_scale=True, unit_divisor=1024
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))

    if expected_md5 and md5sum(save_path) != expected_md5:
        raise RuntimeError(f"MD5 校驗失敗：{save_path}")

    return save_path

def extract_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)

def download_dataset(name: str, root: str = './data'):
    root = Path(root)
    spec = DATASETS[name]
    for split, url in spec['urls'].items():
        save_path = root / name / f'{split}.zip'
        md5 = spec['md5'].get(split)
        try:
            download_file(url, save_path, md5)
            extract_zip(save_path, root / name / split)
            print(f"[OK] {name}/{split} 完成")
        except Exception as e:
            print(f"[FAIL] {name}/{split}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--root', default='./data')
    args = parser.parse_args()
    download_dataset(args.dataset, args.root)