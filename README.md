# CS6886W – VGG6 on CIFAR‑10 (Colab + Local)
**Repo:** https://github.com/mahendren84/CS6886W_CS24M523_A1

Minimal, modular, and fully reproducible starter for the 5‑part assignment:

- **Q1**: Baseline training with CIFAR‑10 **normalization + augmentations**, report **final test accuracy** and **train/val curves**.  
- **Q2**: Compare **activations**, **optimizers**, and vary **batch size / epochs / learning rate** via **W&B Bayesian sweeps**.  
- **Q3**: Provide **parallel‑coordinates**, **validation accuracy vs step** (scatter), and **train/val curves** (exported from W&B and/or script‑generated).  
- **Q4**: Identify and **verify the single best validation** configuration; save the **stored model** and evaluate with **Test.py**.  
- **Q5**: Clean, modular repo with exact **commands** and **seed**. *(Compression module **not** required.)*

---
Python version used (same as collab)
Python 3.12.12
---

## 1) Repository layout (expected)

> If any file is missing locally, create it as per your assignment scaffold.

```
CS6886W_CS24M523_A1/
├─ DataModel.py          # Q1a: CIFAR-10 transforms + loaders (normalization & augmentations)
├─ Model.py              # Q1/Q2a: VGG6 (pluggable activation)
├─ Train.py              # Q1b/Q1c + Q2 + Q3 + Q5: training, W&B logging, checkpointing
├─ Test.py               # Q4: load stored model and print test accuracy
├─ sweep.yaml            # Q2/Q3a: Bayesian sweep configuration
├─ scripts/
│  ├─ run_baseline.sh    # Q1 baseline run
│  ├─ run_sweep.sh       # Q2 sweep creation + agent instructions
│  ├─ test_model.sh      # Q4 evaluate stored best model
│  └─ make_plots.sh      # Q3 export plots to ./plots/
└─ checkpoints/          # saved model(s) (best.pt) – created by Train.py
```

---

## 2) Reproducibility defaults

- **Seed:** `42` (model, data split, CUDA)  
- **Determinism:** cuDNN deterministic ON; benchmark OFF  
- **CIFAR‑10 normalization:**  
  - `mean = (0.4914, 0.4822, 0.4465)`  
  - `std  = (0.2470, 0.2435, 0.2616)`  
- **Augmentations (train):** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `RandomRotation(15)`  
  *(Optional switch:* `--autoaugment` → `AutoAugment(CIFAR10)`)*

---

## 3) Quick start – **two official paths**

### A) Google Colab (recommended)

**Step 0 – Clone the repo and enter it**
```bash
%cd /content
!git clone https://github.com/mahendren84/CS6886W_CS24M523_A1.git
%cd CS6886W_CS24M523_A1
```

**Step 1 – Install packages & login to W&B**
```bash
!pip install torch torchvision tqdm wandb numpy rich pandas matplotlib
import wandb; wandb.login()   # follow the link → paste API key
```

**Step 2 – Make scripts executable (once)**
```bash
!chmod +x scripts/*.sh
```

**Q1 — Baseline training (one run)**
```bash
!bash scripts/run_baseline.sh
```
- Logs **train/val curves** to W&B  
- Saves **best checkpoint** to `checkpoints/best.pt`  
- Prints final **test accuracy**

**Q2/Q3a — Sweeps**
- **If you already have runs** (existing sweeps): skip to **Q3 plots** below.  
- **Create a new sweep (Bayes, 20+ trials):**
  ```bash
  !bash scripts/run_sweep.sh
  # copy printed SWEEP path: <entity>/cs6886-vgg6-cifar10/<SWEEP_ID>
  !wandb agent --count 20 <entity>/cs6886-vgg6-cifar10/<SWEEP_ID>
  ```

**Q3 — Export required plots (PNG)**
- **From the W&B UI** (recommended):  
  - **Parallel‑coordinates**: Project → *Add Chart* → *Parallel Coordinates*; axes: `activation`, `optimizer`, `batch_size`, `epochs`, `lr`, `weight_decay`; color: `val_acc` → **Export** (PNG)  
  - **Val‑Acc vs Step**: pick a run (best/representative) → export `val_acc_vs_step` scatter  
  - **Train/Val curves**: export `train_loss`, `val_loss`, `train_acc`, `val_acc`

**Q4 — Evaluate stored model**
```bash
!bash scripts/test_model.sh
```
- Loads `checkpoints/best.pt` and prints **test accuracy**


---

### B) Local / VS Code (Windows/Mac/Linux)

> Windows venvs may use `.\.venv\Scripts\python` or `.\.venv\bin\python`. Use whichever exists.

**Step 0 – Clone and enter repo**
```bash
git clone https://github.com/mahendren84/CS6886W_CS24M523_A1.git
cd CS6886W_CS24M523_A1
```

**Step 1 – Create & use a virtual environment**

**Windows (no activation path – simplest):**
```powershell
# Create venv if missing
python -m venv .venv

# Install PyTorch FIRST from official index:
.\.venv\bin\pip install --upgrade pip wheel setuptools
.\.venv\bin\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (Or CUDA 12.1 build if you have compatible NVIDIA drivers)
# .\.venv\bin\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest:
.\.venv\bin\pip install tqdm wandb numpy rich pandas matplotlib
```

**Mac/Linux (or Windows with activation):**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Remaining deps
pip install tqdm wandb numpy rich pandas matplotlib
```

**Step 2 – Login to W&B**
```powershell
# Windows
.\.venv\bin\python - << 'PY'
import wandb; wandb.login()
PY
```
```bash
# Mac/Linux
python - << 'PY'
import wandb; wandb.login()
PY
```

**Q1 — Baseline training**
```powershell
# Windows
.\.venv\bin\python Train.py --project cs6886-vgg6-cifar10 --activation relu --optimizer sgd --lr 0.1 --batch_size 128 --epochs 100 --scheduler cosine --seed 42
```
```bash
# Mac/Linux
python Train.py --project cs6886-vgg6-cifar10 --activation relu --optimizer sgd --lr 0.1 --batch_size 128 --epochs 100 --scheduler cosine --seed 42
```

**Q2/Q3a — Sweeps (use existing runs or create new)**
```powershell
wandb sweep sweep.yaml
# then run exactly 20 trials:
wandb agent --count 20 <entity>/cs6886-vgg6-cifar10/<SWEEP_ID>
```

**Q4 — Evaluate stored model**
```powershell
# Windows
.\.venv\bin\python Test.py --ckpt checkpoints\best.pt --batch_size 128
```
```bash
# Mac/Linux
python Test.py --ckpt checkpoints/best.pt --batch_size 128
```

---

## 4) CLI knobs you can change (Train.py)

- `--activation`: `relu | silu | gelu | tanh | sigmoid`  
- `--optimizer` : `sgd | sgd_nesterov | adam | adagrad | rmsprop | nadam`  
- `--lr` : float (e.g., `0.1`, `0.01`)  
- `--batch_size` : `64 | 128 | 256 | …`  
- `--epochs` : `50 | 100 | …`  
- `--scheduler` : `cosine | step | plateau | none`  
- `--weight_decay`, `--momentum`, `--gamma`, `--step_size`, `--autoaugment`

Example:
```bash
python Train.py \
  --activation silu \
  --optimizer sgd_nesterov \
  --lr 0.05 \
  --batch_size 128 \
  --epochs 100 \
  --scheduler cosine \
  --seed 42
```

---

## 5) Using existing runs only (no reruns)

- Generate all plots from W&B **without training**:
  ```bash
  ```
- (Optional) Download the **existing** stored model from W&B Files tab via API:
  ```python
  import wandb, os
  wandb.login()
  ENTITY="<YOUR_WANDB_ENTITY>"; PROJECT="cs6886-vgg6-cifar10"
  api = wandb.Api()
  runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.summary and "val_acc" in r.summary]
  best = max(runs, key=lambda r: r.summary["val_acc"])
  os.makedirs("downloads", exist_ok=True)
  for f in best.files():
      if f.name.endswith(("best.pt","best.pth")):
          f.download(root="downloads", replace=True)
  ```

---

## 6) Deliverables (PDF checklist)

- **Q1a**: Augmentations + normalization stats.  
- **Q1b/c**: Baseline hyper‑params; final **test top‑1 accuracy**; **train/val curves**.  
- **Q2**: Discuss differences across **activations**, **optimizers**, and effects of **batch/epochs/LR** (use your sweep runs).  
- **Q3**: Attach `parallel_coordinates.png`, `val_acc_vs_step.png`, `curves_loss.png`, `curves_acc.png` (W&B exports or `plots/`).  
- **Q4**: One best validation configuration (exact values) and `Test.py` output using your stored model (e.g., `checkpoints/best.pt`).  
- **Q5**: Link to this GitHub repo; seed (42), environment info, and the commands above.

---

## 7) Troubleshooting & FAQ

- **PyTorch “No matching distribution” on Windows** → install torch from the **official PyTorch index** (CPU or CUDA):  
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- **AMP deprecation warnings** (newer PyTorch):  
  Replace `from torch.cuda.amp import autocast, GradScaler` with `from torch.amp import autocast, GradScaler`, and use:  
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  device_type = "cuda" if device == "cuda" else "cpu"
  scaler = GradScaler(device_type=device_type, enabled=(device_type=="cuda"))
  with autocast(device_type, enabled=(device_type=="cuda")):
      ...
  ```
- **Sweep won’t stop** → run agent with `--count 20` to cap trials.  
- **Dataloaders slow on Windows** → run with `--num_workers 0`; on Colab/Linux use `2–4`.  
- **Minor variance across machines** is normal; keep seed, batch size, and GPU similar when comparing.

---

## 8) License & citation

- Dataset: **CIFAR‑10** (auto‑downloaded via `torchvision`).  
- Code: intended for coursework; adapt with attribution.
