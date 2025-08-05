import torch
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
import random
import numpy as np

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    valid_loss_min = checkpoint["valid_loss_min"]
    return model, optimizer, checkpoint["epoch"], valid_loss_min.item()

def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def loss_func(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    return {
        f"Micro-F1: {f1_score(y_true, y_pred_bin, average="micro", zero_division=0):.4f} | "
        f"Macro-F1: {f1_score(y_true, y_pred_bin, average="macro", zero_division=0):.4f} | "
        f"Precision: {precision_score(y_true, y_pred_bin, average="micro", zero_division=0):.4f} | "
        f"Recall: {recall_score(y_true, y_pred_bin, average="micro", zero_division=0):.4f} | "
    }


