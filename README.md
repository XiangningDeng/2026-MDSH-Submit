# MIND News Recommendation Experiments

This project contains notebook-based experiments for multiple NewsRec models on the MIND dataset:

- `NRMS` (`nrms_MIND.ipynb`)
- `NAML` (`naml_MIND.ipynb`)
- `NPA` (`npa_MIND.ipynb`)
- `LSTUR` (`lstur_MIND.ipynb`)
- `DKN` (`dkn_MIND.ipynb`)

It includes local preprocessing via `prepare_data.py` and a vendored `recommenders` package used by the notebooks.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── prepare_data.py
├── nrms_MIND.ipynb
├── naml_MIND.ipynb
├── npa_MIND.ipynb
├── lstur_MIND.ipynb
├── dkn_MIND.ipynb
├── recommenders/
│   └── ... model and data utilities used by notebooks
├── utils/
│   ├── *.yaml                 # tracked config files
│   └── *.pkl, *.npy           # generated artifacts (ignored by git)
└── data/                      # local dataset folder (ignored by git)
    ├── train/
    │   ├── news.tsv
    │   └── behaviors.tsv
    └── valid/
        ├── news.tsv
        └── behaviors.tsv
```

Important:

- `data/` is intentionally not uploaded to GitHub.
- generated files in `utils/` (for example `*.pkl`, `*.npy`) are also not uploaded.
- after cloning, users should create `data/` locally and run `python prepare_data.py`.

## 1) Environment Setup

From project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: macOS Apple Silicon GPU acceleration

If you use Apple Silicon (M1/M2/M3), you can install Metal acceleration:

```bash
pip install tensorflow-macos tensorflow-metal
```

For Linux/Windows users, skip this step.

### Optional: NVIDIA GPU acceleration (Linux/Windows)

If you use an NVIDIA GPU, install a CUDA-compatible TensorFlow build.
Use the TensorFlow + CUDA versions that match your local NVIDIA driver and CUDA runtime.

A common setup flow is:

1. Install/update NVIDIA driver
2. Install CUDA and cuDNN versions compatible with your TensorFlow version
3. Install TensorFlow in your environment

You can verify GPU visibility with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 2) Prepare Data Folder

Expected structure:

```text
data/
  train/
    news.tsv
    behaviors.tsv
  valid/
    news.tsv
    behaviors.tsv
```

`prepare_data.py` reads from:

- `data/train/news.tsv`
- `data/train/behaviors.tsv`
- `data/valid/news.tsv`

## 3) Generate Utils Artifacts

Run:

```bash
python prepare_data.py
```

What this does:

- Builds dictionaries from your local `data/` files
- Tries to download official MIND utility resources from Hugging Face
- Falls back to random embedding initialization if download fails
- Writes artifacts into `utils/` (for example `*.pkl`, `*.npy`)

Note: generated artifacts in `utils/` are intentionally ignored by `.gitignore`.

## 4) Run Notebooks

Start Jupyter:

```bash
jupyter lab
```

Then open any of:

- `nrms_MIND.ipynb`
- `naml_MIND.ipynb`
- `npa_MIND.ipynb`
- `lstur_MIND.ipynb`
- `dkn_MIND.ipynb`

## 5) Reproducibility Notes

- Commit source code, configs (`utils/*.yaml`), and notebooks.
- Do not commit generated data/artifacts (`data/`, `utils/*.pkl`, `utils/*.npy`, checkpoints).
- If someone clones this repo, they can reproduce by:
  1. Installing dependencies
  2. Placing MIND files under `data/`
  3. Running `python prepare_data.py`
  4. Running notebooks

