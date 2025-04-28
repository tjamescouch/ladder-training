# Curriculum‑Ladder Training

**Fast progressive‑resolution training for image classification, with optional Batch‑Norm freezing**

---

## Overview

This repo contains a minimal, reproducible reference implementation of the *curriculum‑ladder* strategy we have been exploring in our notebooks:

- **Baseline** – conventional fixed‑resolution training (96×96 pixels).
- **Ladder** – start at 32×32 and step through 40 → 56 → 72 → 96 px with a short epoch budget per rung.
- **Ladder + BN‑freeze** – same curriculum, but freezes running‐stat updates in all BatchNorm layers once the network reaches its target accuracy on a rung.\
  *Result: **\~1.8 × faster wall‑clock to 60 % validation accuracy** on STL‑10.*

```
mean Δ‑time  : 2.82 ± 0.58 min (95 % CI)  ← baseline – ladder+BN
speed‑up     : 1.8×                (10 seeds: 31‑40)
```

> *If you are reviewing the repo for the first time, jump to the* **Quick Start** *section below – you can reproduce the table above with one command.*

---

## Folder layout

```
.
├── notebooks/          ← exploratory & reproducibility notebooks
│   ├── stl10_ladder_demo.ipynb
│   └── …
├── data/                ← kept **out** of git, see `.gitignore`
└── README.md            ← you are here
```

A full `.gitignore` keeps binary datasets, Jupyter checkpoints & virtual‑env junk out of the repository.

---

## Quick Start

### 1. Environment

```bash
conda create -n ladder python=3.11 pytorch torchvision torchaudio -c pytorch
conda activate ladder
pip install -r requirements.txt   # lightning, tqdm, seaborn, etc.
```

### 2. Get a dataset (STL‑10 for the table above)

```bash
python -c "import torchvision; torchvision.datasets.STL10(root='data', split='train', download=True)"
```

> **Tip** – all notebooks/scripts look for datasets in the **`data/`** directory by default.

### 3. Reproduce our 10‑seed experiment

See the stl10_ladder_demo.ipyn notebook.

The script prints per‑epoch logs and ends with a paired‑t summary identical to the block in this README.

---

## How it works  *(short version)*

1. **Progressive resolution** keeps the early convolution layers in their comfort zone (small receptive fields) and offers bigger batches → better hardware utilisation.
2. **Adaptive exit** stops training the moment the validation accuracy on the current rung crosses the 60 % threshold, instead of wasting time on diminishing returns.
3. **Batch‑Norm freezing** cuts the extra forward pass cost when images grow larger by skipping the running‑stat updates (i.e. `train()` → `eval()` on BN modules only).

Full details and ablation results are in `STL10_ladder_BNfreeze.ipynb`.

---

## Results summary

| Setting                | Resolution schedule                        | Mean wall‑clock time to 60 % | Speed‑up |
| ---------------------- | ------------------------------------------ | ---------------------------- | -------- |
| **Baseline**           | 96 px for 40 epochs                        | 6.5 ± 0.9 min                | 1.0×     |
| **Ladder**             | 32 → 40 → 56 → 72 → 96 (5/5/3/2/20 epochs) | 3.9 ± 0.7 min                | **1.7×** |
| **Ladder + BN‑freeze** | same, BN stats frozen per rung             | **3.6 ± 0.6 min**            | **1.8×** |

*(Numbers averaged over seeds 31‑40 on STL‑10; see notebooks for per‑seed details.)*


---

## Citing or re‑using this code

If you build on this repository, please cite:

```bibtex
@misc{couch2025ladder,
  title  = {Curriculum Ladder Training with Batch‑Norm Freezing},
  author = {James Couch and contributors},
  year   = {2025},
  howpublished = {GitHub},
  url    = {https://github.com/<org>/<repo>}
}
```

---

## Licence

This project is released under the **MIT License** (see `LICENSE`).\
STL‑10, CIFAR‑10 and other datasets remain under their respective original licences/terms of use and are **not** redistributed here – the download scripts pull them from their canonical mirrors.

---

## Acknowledgements

- Yann LeCun, Corinna Cortes & Chris Burges for MNIST (public domain).
- Adam Coates, Andrew Ng & Honglak Lee for STL‑10.
- The PyTorch & Torchvision teams.
- The fast.ai community for early inspiration on progressive resizing.

---

**Happy training!**

