"""Reproducibility, batching, metrics, and serialization utilities."""

import copy
import json
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_batch(batch, device):
    text = batch['text'].to(device).transpose(0, 1)
    vision = batch['vision'].to(device).transpose(0, 1)
    audio = batch['audio'].to(device).transpose(0, 1)
    qmask = batch['qmask'].to(device)
    input_mask = batch['umask'].to(device)
    labels = batch['label'].to(device).view(-1, 1)
    output_mask = torch.ones_like(labels, dtype=torch.float32)
    lengths = input_mask.sum(dim=1).long().tolist()
    return (
        text, vision, audio, qmask, input_mask,
        labels, output_mask, lengths,
    )

def classification_metrics(labels, predictions):
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'weighted_f1': float(
            f1_score(labels, predictions, average='weighted', zero_division=0)
        ),
        'macro_f1': float(
            f1_score(labels, predictions, average='macro', zero_division=0)
        ),
    }

def format_metrics(metrics):
    return ', '.join(
        '{}: {:.4f}'.format(name, value)
        for name, value in metrics.items()
    )

def clone_state_dict(model):
    return copy.deepcopy(model.state_dict())

def save_summary(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
