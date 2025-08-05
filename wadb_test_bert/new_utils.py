import torch
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
from new_config import TARGET_LIST, DEVICE, TRAIN_DATASET_PATH

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

def compute_metrics(y_true, y_pred, threshold=0.6):

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

def get_class_weights(df):

    label_counts = label_count(df)

    imbalanced = list(zip(label_counts['label'], label_counts['occurrences']))

    total_samples = sum(count for label, count in imbalanced)
    num_classes = len(imbalanced)

    class_weights = {}
    for label, count in imbalanced:
        weight = total_samples / (num_classes * count)
        class_weights[label] = weight

    weights_list = [class_weights[label] for label, _ in imbalanced]

    pos_weight = torch.tensor(weights_list, dtype=torch.float, device=DEVICE)

    return pos_weight

def get_pos_weights(path):

    train_df = pd.read_csv(path)

    pos_weight = get_class_weights(train_df)

    print(pos_weight)

    return pos_weight
