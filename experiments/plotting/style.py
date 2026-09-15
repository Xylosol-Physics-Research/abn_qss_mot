# experiments/plotting/style.py
"""
論文圖表統一樣式：NeurIPS / ICLR 風格
"""
import matplotlib.pyplot as plt

# 全局配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# 顏色方案（色盲友善）
COLORS = {
    'hungarian': '#1f77b4',   # 藍
    'greedy':    '#ff7f0e',   # 橙
    'nullspace': '#2ca02c',   # 綠
    'central':   '#d62728',   # 紅
    'distributed': '#9467bd',  # 紫
    'zsp':       '#2ca02c',
    'baseline':  '#7f7f7f',   # 灰
}

# 標記
MARKERS = {
    'hungarian': 'o',
    'greedy': 's',
    'nullspace': '^',
    'central': 'D',
    'distributed': 'v',
}