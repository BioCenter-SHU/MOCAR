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

The original datasets and pre-extracted features are not redistributed in this repository. For reproducing the reported results, use the pre-extracted feature packages listed below. The original-dataset links are provided separately for licensing, citation, and users who wish to extract their own features.

### Pre-extracted Features Used by MOCAR

| Task | Download | Required local filename | Compatibility notes |
|---|---|---|---|
| IEMOCAP-4 | [Google Drive](https://drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1u1efdbBV3HP8FLj3Gy1bvQ) (code: `ipnv`) | `iemocap_multi_features_4.pkl` | GraphSmile-format feature file used by the current loader. |
| IEMOCAP-6 | [Google Drive](https://drive.google.com/drive/folders/1TwT9z6N6SJadsVkDhSNVBiF9ZygEyA6l) | `iemocap_multimodal_features.pkl` | The downloaded feature file matches the current MOCAR loader. |
| MELD | [Google Drive](https://drive.google.com/drive/folders/1TwT9z6N6SJadsVkDhSNVBiF9ZygEyA6l) | `meld_multimodal_features.pkl` | The downloaded feature file matches the current MOCAR loader. |
| CMU-MOSEI7 | [Google Drive](https://drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1u1efdbBV3HP8FLj3Gy1bvQ) (code: `ipnv`) | `cmumosei_multi_regression_features.pkl` | GraphSmile-format regression feature file; MOCAR converts its labels into seven classes. |
| CH-SIMS2.0 | [Google Drive](https://drive.google.com/drive/folders/1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/13Ds2_XDIGUqMHt4lXNLQSQ) (code: `icmi`) | `sims2_unaligned_001.pkl` | The official package may provide the unaligned data as `SimsLargeV6.pkl`. Rename it or convert it to the filename and split/feature schema expected by `src/dataloader.py`. |

The IEMOCAP-4 and CMU-MOSEI7 links are published by the [GraphSmile repository](https://github.com/lijfrankopen/GraphSmile). The IEMOCAP-6 and MELD files are preprocessed multimodal features derived from the feature format used by [SDT](https://github.com/butterfliesss/SDT).

### Original Dataset Sources

| Dataset | Official source | Notes |
|---|---|---|
| IEMOCAP | [USC SAIL IEMOCAP](https://sail.usc.edu/iemocap/) | Access requires accepting the dataset license/request procedure. |
| MELD | [MELD website](https://affective-meld.github.io/) and [official repository](https://github.com/declare-lab/MELD) | Contains the original conversational emotion data and documentation. |
| CMU-MOSEI | [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) | Provides official dataset access and computational-sequence tools. |
| CH-SIMS v2.0 | [official repository](https://github.com/thuiar/ch-sims-v2) and [dataset website](https://thuiar.github.io/sims.github.io/chsims) | Use the supervised-data package for the MOCAR binary task. |

Raw audio/video datasets cannot be passed directly to the training scripts. They must first be converted to the pickle schemas expected by `src/dataloader.py`. The links in the first table are therefore recommended when reproducing the released experiments.

### File Placement

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
