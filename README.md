# abn_qss_mot
本文系統評估零空間投影（ZSP）在三類感知任務上的表現： 單幀 MOT、多幀 MOT、協同感知。結果顯示 ZSP 在線性分配 問題上無法超越匈牙利算法。我們分析根本原因，並劃定 ZSP 的適用邊界。
# ABN-QSS MOT: Zero-Space Projection for Multi-Object Tracking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/main.pdf)

> **A systematic negative result on physics-native computing for perception tasks.**

This repository contains the complete code, data, and figures for the paper:

**"The Boundaries of Physics-Native Computing: A Systematic Negative Result on Zero-Space Projection for Multi-Object Tracking and Cooperative Perception"**

Lien-Hsing Chang, ABN-QSS Research Team — September 2026

---

## 📋 TL;DR

We evaluated **Zero-Space Projection (ZSP)** — the core mechanism of the ABN-QSS v3.0 physics-native computing architecture — on three perception tasks. **All three experiments produced negative results:**

| Task | Hungarian | ZSP | Verdict |
|------|-----------|-----|---------|
| **Single-frame MOT** | 100% / 0.01 ms | 99.96% / 34.56 ms | ❌ 3,456× slower |
| **Multi-frame MOT** | 90.9% | 84.2% | ❌ −6.7% (p=0.0001) |
| **Cooperative perception** | 100% / 0.03 ms / 0 B | 100% / 15.99 ms / 480 B | ❌ Slower + comm. cost |

**Root cause:** Linear assignment problems have a *dual structure* that enables the Hungarian algorithm to achieve **exact optimality in polynomial time**. ZSP is an iterative approximation and cannot, by construction, outperform the exact algorithm.

**Conclusion:** ZSP's value lies in solving problems **without** good algorithms (material simulation, physics-layer cryptography, high-precision sensing), not in replacing problems already solved by exact methods.

---

## 🎯 Key Contributions

1. **Complete experimental framework** spanning digital-twin verification to real benchmarks
2. **Systematic negative results** on three perception tasks
3. **Root-cause analysis** via LP duality theory
4. **Explicit applicability boundary** for physics-native computing
5. **Fully open-source** code, data, and 8 paper figures

---

## 📁 Repository Structure

```
abn_qss_mot/
├── core/                       # Core algorithms
│   ├── __init__.py
│   ├── constraint.py           # Constraint matrix construction
│   ├── nullspace.py            # SVD-based null-space extraction
│   ├── solver.py               # Projected gradient solver
│   ├── multi_frame.py          # Multi-frame extension
│   └── cooperative.py          # Cooperative perception extension
│
├── data/                       # Dataset loaders
│   ├── __init__.py
│   ├── download.py             # Dataset download scripts
│   ├── preprocess.py           # Feature extraction (ReID, boxes)
│   └── dataset.py              # PyTorch Dataset wrappers
│
├── experiments/                # Benchmarks and experiments
│   ├── benchmark.py            # Single-frame MOT benchmark
│   ├── benchmark_multiframe.py # Multi-frame MOT benchmark
│   ├── cooperative_benchmark.py# Cooperative perception benchmark
│   ├── statistical_significance.py
│   ├── noise_robust.py         # Noise resilience test
│   └── plotting/               # Figure generation
│       ├── style.py            # Matplotlib style config
│       ├── fig1_digital_twin.py
│       ├── fig2_single_frame.py
│       ├── fig3_multiframe.py
│       ├── fig4_cooperative.py
│       ├── fig5_noise.py
│       ├── fig6_convergence.py
│       ├── fig7_scaling.py
│       ├── fig8_boundary.py
│       └── plot_all.py         # One-click figure generation
│
├── tests/                      # Unit tests (13 tests, 100% pass)
│   ├── test_constraint.py
│   ├── test_nullspace.py
│   └── test_solver.py
│
├── eval/                       # MOT metrics
│   ├── __init__.py
│   └── metrics.py              # MOTA, IDF1, ID Switch
│
├── figures/                    # Generated figures (PDF + PNG)
│   ├── fig1_digital_twin.pdf
│   ├── fig2_single_frame.pdf
│   ├── fig3_multiframe.pdf
│   ├── fig4_cooperative.pdf
│   ├── fig5_noise.pdf
│   ├── fig6_convergence.pdf
│   ├── fig7_scaling.pdf
│   └── fig8_boundary.pdf
│
├── paper/                      # LaTeX manuscript
│   ├── main.tex
│   ├── sections/
│   ├── appendix/
│   ├── refs.bib
│   └── Makefile
│
├── results/                    # Raw experimental outputs
│   ├── single_frame.txt
│   ├── multi_frame.txt
│   ├── cooperative.txt
│   └── statistical.txt
│
├── pyproject.toml              # Package configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- NumPy 1.24+
- SciPy 1.10+
- (Optional) PyTorch 2.0+ for ReID feature extraction

### Install from source

```bash
git clone https://github.com/<your-username>/abn_qss_mot.git
cd abn_qss_mot
pip install -e .
```

### Install test dependencies

```bash
pip install -e ".[dev]"
# or
pip install pytest pytest-cov
```

---

## 🚀 Quick Start

### 1. Run unit tests

```bash
pytest tests/ -v --cov=core
```

Expected: **13 passed** in ~0.5s.

### 2. Verify digital-twin claims

```bash
python -m experiments.n4_verify
```

Expected output:
```
N=3: D=8,  K_theory=3,  K_actual=3,  match=True
      投影線性度 R²=1.000000, 斜率=0.0075
N=4: D=16, K_theory=6,  K_actual=6,  match=True
      投影線性度 R²=1.000000, 斜率=0.0080
...
```

### 3. Run single-frame MOT benchmark

```bash
python -m experiments.benchmark
```

Expected output:
```
T,D      方法           時間(ms)         成本           步數
------------------------------------------------------------
10       hungarian      0.01±0.00   -8.6026      0
10       greedy         0.05±0.04   -7.9484      0
10       nullspace      34.56±1.22  -8.5892    500
...
```

### 4. Run multi-frame MOT benchmark

```bash
python -m experiments.benchmark_multiframe
```

### 5. Run cooperative perception benchmark

```bash
python -m experiments.cooperative_benchmark
```

### 6. Run statistical significance test

```bash
python -m experiments.statistical_significance
```

Expected output:
```
匈牙利:      0.909 ± 0.122
多幀零空間:  0.842 ± 0.129
平均差異:    -0.067 ± 0.079
t 統計量:    -4.580
p 值:        0.0001  ***
多幀勝出比例: 0.0%
```

### 7. Regenerate all figures

```bash
python -m experiments.plotting.plot_all
```

All 8 figures will be saved to `figures/`.

---

## 📊 Reproducing Paper Results

| Paper Section | Command | Expected Output |
|---------------|---------|-----------------|
| §4.1 Single-frame | `python -m experiments.benchmark` | Table 1, Figure 2 |
| §4.2 Multi-frame | `python -m experiments.benchmark_multiframe` | Table 2, Figure 3 |
| §4.3 Cooperative | `python -m experiments.cooperative_benchmark` | Table 3, Figure 4 |
| §4.4 Statistical | `python -m experiments.statistical_significance` | Table 2 |
| §4.5 Noise | `python -m experiments.noise_robust` | Figure 5 |
| §4.6 Digital twin | `python -m experiments.n4_verify` | Figure 1 |
| §5 Analysis | `python -m experiments.plotting.fig6_convergence` | Figure 6 |
| §5.3 Scaling | `python -m experiments.plotting.fig7_scaling` | Figure 7 |

All figures can be regenerated with a single command:

```bash
python -m experiments.plotting.plot_all
```

---

## 🔬 Digital Twin Verification

The ABN-QSS v3.0 architecture claims two theoretical results:

1. **Projection linearity**: $R^2 = 1.000000$
2. **Null-space dimension**: $K = N(N-1)/2$

Our independent implementation reproduces both exactly:

```python
from core.nullspace import NullSpaceExtractor
from scipy.linalg import hadamard

N = 4
D = 2 ** N
K = N * (N - 1) // 2
H = hadamard(D) / np.sqrt(D)
M = H[K:, :]

extractor = NullSpaceExtractor(M)
print(f"Null-space dimension: {extractor.K}")  # Output: 6
print(f"Expected: {K}")                        # Output: 6
```

See `experiments/n4_verify.py` for the complete verification.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{chang2026boundaries,
    title={The Boundaries of Physics-Native Computing: A Systematic 
           Negative Result on Zero-Space Projection for Multi-Object 
           Tracking and Cooperative Perception},
    author={Chang, Lien-Hsing and {ABN-QSS Research Team}},
    journal={https://orcid.org/0009-0002-1476-6341},
    year={2026}
}
```

---

## 🤝 Contributing

We welcome contributions, especially:

- **Additional benchmarks**: test ZSP on other problem classes (non-linear constraint satisfaction, material simulation, etc.)
- **Alternative solvers**: L-BFGS, ADMM, or second-order methods
- **Hardware mapping**: Verilog-A / SPICE models for FPGA/ASIC
- **Bug reports**: open an issue with a minimal reproducible example

### Development workflow

```bash
# Fork and clone
git clone https://github.com/<your-username>/abn_qss_mot.git

# Create feature branch
git checkout -b feature/your-feature

# Run tests before committing
pytest tests/ -v

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature

# Open a Pull Request
```

---

## 📜 License

- **Code**: MIT License — see [LICENSE](LICENSE)
- **Data & Figures**: CC-BY-4.0
- **Paper**: arXiv non-exclusive license

---

## 🙏 Acknowledgments

This work builds on the ABN-QSS v3.0 theoretical framework. We thank the ABN-QSS research team for the digital-twin specifications, and the open-source community for SciPy, NumPy, and Matplotlib.

We are especially grateful to reviewers and colleagues who encouraged us to publish this negative result — **negative results are knowledge too.**

---

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/Xyloso-Physics-Research/abn_qss_mot/issues)
- **Email**: LandsingChang@gmail.com
- **Paper**: [ORCiD](https://orcid.org/0009-0002-1476-6341)

---

## 🗺️ Roadmap

### ✅ Completed (v0.1.0)

- [x] Digital-twin verification (R²=1.0, K=N(N-1)/2)
- [x] Single-frame MOT benchmark
- [x] Multi-frame MOT benchmark with soft continuity
- [x] Cooperative perception benchmark
- [x] Statistical significance test (30 seeds)
- [x] 8 paper figures
- [x] 13 unit tests (100% pass)
- [x] LaTeX manuscript

### 🚧 In Progress (v0.2.0)

- [ ] Real dataset evaluation (Griffin, TUMTraf)
- [ ] Noise resilience experiments
- [ ] L-BFGS / ADMM alternative solvers
- [ ] Verilog-A models for FPGA mapping

### 🔮 Planned (v0.3.0+)

- [ ] Material simulation benchmark
- [ ] Physics-layer cryptography prototype
- [ ] High-precision sensing experiments
- [ ] Silicon photonics tape-out (partnership TBD)

---

## ⭐ Star History

If this negative result saves you time, please consider giving it a star ⭐ — it helps others find this work.

---

<div align="center">

**"Knowing when *not* to use a tool is as important as knowing when to use it."**

</div>
