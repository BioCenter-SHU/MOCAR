# MOCAR: Confusion-Aware Multimodal Emotion and Sentiment Recognition

This repository contains the official implementation of **MOCAR**. The model maintains modality-wise online confusion matrices and combines confusion-aware regularization with hard class-pair margin learning to improve the recognition of easily confused categories.

The released code supports the main experiments on **IEMOCAP-4**, **IEMOCAP-6**, **MELD**, **CMU-MOSEI7**, and binary **CH-SIMS2.0**. Each task has a dedicated entry point whose default arguments reproduce the corresponding main-experiment configuration. Ablation studies, plotting utilities, and hyperparameter-search code are intentionally excluded.

## 📂 Project Structure

```text
.
├── data/
│   └── .gitkeep                       # Place pre-extracted feature files here
├── outputs/
│   ├── iemocap4/
│   ├── iemocap6/
│   ├── meld/
│   ├── cmumosei7/
│   └── sims2/                         # Checkpoints, training logs, and evaluation logs
├── src/
│   ├── confusion_regularizer.py      # MOCAR confusion-aware objectives
│   ├── dataloader.py                 # Dataset readers and collate functions
│   ├── model.py                      # Core multimodal network
│   ├── runtime.py                    # Logging and evaluation-only runtime helpers
│   ├── sims_binary_training.py       # CH-SIMS2.0 training loop
│   └── sims_classification_utils.py  # CH-SIMS2.0 metrics and utilities
├── train_iemocap4.py                  # IEMOCAP four-class entry point
├── train_iemocap6.py                  # IEMOCAP six-class entry point
├── train_meld.py                      # MELD entry point
├── train_cmumosei7.py                 # CMU-MOSEI seven-class entry point
├── train_sims2.py                     # Binary CH-SIMS2.0 entry point
├── environment.yml                    # Conda environment definition
├── requirements.txt                   # Pip dependencies
└── README.md
```

Each task directory under `outputs/` uses one consistent layout:

```text
outputs/<task>/
├── checkpoints/  # Best checkpoint selected during training
└── logs/         # Timestamped training and evaluation logs
```

## 🛠️ Environment Setup

The code was verified on Ubuntu 20.04 with an NVIDIA GeForce RTX 3090, CUDA 11.8, Python 3.8.20, and PyTorch 2.0.0.

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate mocar
```

### 2. Alternative Pip Installation

Create and activate a Python 3.8 environment, then run:

```bash
python -m pip install -r requirements.txt
```

The dependency set includes:

- PyTorch 2.0.0 with CUDA 11.8
- NumPy 1.24.4
- SciPy 1.10.1
- pandas 2.0.3
- scikit-learn 1.3.2

## 💾 Data Preparation

The original datasets and pre-extracted features are not redistributed in this repository. Obtain each dataset under its license and prepare a pickle file compatible with the corresponding dataset class in `src/dataloader.py`.

Place the files in `data/` using the following names:

| Task | Feature file | Text / vision / audio dimensions | Classes |
|---|---|---:|---:|
| IEMOCAP-4 | `data/iemocap_multi_features_4.pkl` | 1024 / 512 / 100 | 4 |
| IEMOCAP-6 | `data/iemocap_multimodal_features.pkl` | 1024 / 342 / 1582 | 6 |
| MELD | `data/meld_multimodal_features.pkl` | 1024 / 342 / 300 | 7 |
| CMU-MOSEI7 | `data/cmumosei_multi_regression_features.pkl` | 1024 / 35 / 74 | 7 |
| CH-SIMS2.0 | `data/sims2_unaligned_001.pkl` | 768 / 177 / 25 | 2 |

You can also keep a feature file outside the repository and pass its path explicitly:

```bash
python train_iemocap6.py --data-path /path/to/iemocap_multimodal_features.pkl
```

CMU-MOSEI continuous sentiment labels are mapped to seven classes using the intervals `<-2`, `[-2,-1)`, `[-1,0)`, `0`, `(0,1]`, `(1,2]`, and `>2`. The binary CH-SIMS2.0 task uses `score <= 0` and `score > 0` as its two classes.

## 🚀 How to Run

All entry points use GPU 0 by default. Running a script without `--checkpoint` starts training with the task-specific configuration embedded in that script. The best checkpoint and a complete timestamped training log are saved automatically.

### Training

```bash
python train_iemocap4.py
python train_iemocap6.py
python train_meld.py
python train_cmumosei7.py
python train_sims2.py
```

To select another GPU:

```bash
python train_meld.py --gpu 1
```

To override a default training argument:

```bash
python train_iemocap6.py --seed 2025 --epochs 100
```

### Checkpoint-Only Evaluation

Passing `--checkpoint` automatically enables evaluation-only mode. The script loads the supplied weights, evaluates the test split, writes a new evaluation log, and performs no training epochs or optimizer updates.

```bash
python train_iemocap4.py --checkpoint outputs/iemocap4/checkpoints/iemocap4_seed61078_best.pt
python train_iemocap6.py --checkpoint outputs/iemocap6/checkpoints/iemocap_seed61078_best.pt
python train_meld.py --checkpoint outputs/meld/checkpoints/meld_seed10068_best.pt
python train_cmumosei7.py --checkpoint outputs/cmumosei7/checkpoints/cmumosei7_seed61078_best.pt
python train_sims2.py --checkpoint outputs/sims2/checkpoints/mocar_sims2_best.pt
```

### Main Arguments

- `--data-path`: Path to the pre-extracted feature pickle.
- `--gpu`: CUDA device index; the default is `0`.
- `--no-cuda`: Run on CPU.
- `--checkpoint`: Load a checkpoint and perform evaluation only.
- `--epochs`: Maximum number of training epochs.
- `--seed` or `--seeds`: Random seed configuration.
- `--save-best-dir`: Directory used for the best checkpoint.
- `--log-dir`: Directory used for timestamped console logs.
- `--no-save-checkpoint`: Train without writing a checkpoint.

Run `python train_<task>.py --help` to inspect all task-specific optimization and MOCAR arguments.

## 📦 Checkpoint Files

The included `.pt` files were trained from scratch with this release. Dialogue-task checkpoints contain the model state, optimizer state, selected epoch, metrics, classification report, confusion matrix, training arguments, and learned MOCAR confusion matrices. The CH-SIMS2.0 checkpoint contains the model state, validation selection information, test metrics, and training arguments.

Checkpoint files larger than 100 MB must be uploaded through Git LFS. The included `.gitattributes` already assigns `outputs/*/checkpoints/*.pt` to Git LFS.

## 📝 Citation and Acknowledgements

If you use this repository, please cite the MOCAR paper. The final citation entry will be added after publication.

Please also cite the original IEMOCAP, MELD, CMU-MOSEI, and CH-SIMS2.0 datasets when using their data or derived features. This implementation retains components from the original multimodal main-experiment codebase; their authors and the maintainers of the public datasets are gratefully acknowledged.
