import datetime
import json
import os
import re
import numpy as np
import torch
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report,precision_recall_curve
import pandas as pd
from new_config import TARGET_LIST, DEVICE, TRAIN_DATASET_PATH, TEXT_COLUMN

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

def compute_metrics(y_true, y_pred, threshold=0.5):

    y_pred_bin = (y_pred >= threshold).astype(int)

    f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred_bin, target_names=TARGET_LIST, zero_division=0)

    return f1_micro, f1_macro, precision, recall, report


def label_count(df):

    # 3. Apply across your DataFrame
    print(df.columns)

    label_cols = [c for c in df.columns if c != 'tool_description']

    # 2) For each row, collect the names of columns where the value is 1
    df['tool_category_list'] = df[label_cols].apply(
        lambda row: [col for col in label_cols if row[col] == 1],
        axis=1
    )

    exploded = df.explode('tool_category_list')

    # 3) Count occurrences of each label
    counts = exploded['tool_category_list'] \
                 .value_counts() \
                 .rename_axis('label') \
                 .reset_index(name='occurrences')

    return counts

def get_pos_weights_from_df(df, label_cols = TARGET_LIST, device = DEVICE):
    # pos_c = count of positives per class
    pos = df[label_cols].sum().astype(float)
    N = len(df)
    neg = N - pos
    # Avoid division by zero; if a class has zero positives, set a big weight
    eps = 1e-8
    pos_weight = (neg / (pos + eps)).values
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)

def get_pos_weights(path, text_col = TEXT_COLUMN, device = DEVICE):
    df = pd.read_csv(path)
    label_cols = [c for c in df.columns if c != text_col]
    return get_pos_weights_from_df(df, label_cols, device)


def save_eval(metrics, report_txt, out_dir = "eval_logs/", run_name = None):

    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"\W+", "_", run_name.strip()) if run_name else "run"

    base = os.path.join(out_dir + run_name)
    txt_path  = base + ".txt"

    # write TXT (metrics + report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Classification Report ===\n")
        f.write(report_txt if report_txt.endswith("\n") else report_txt + "\n")

    print(f"[Saved] {txt_path}")
    return {"txt": txt_path}