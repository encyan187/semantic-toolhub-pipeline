import ast
from gpt_config import TARGET_LIST
from gpt_utils import parse_list, save_eval
from classify_tool import classify_tool
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import pandas as pd
from tqdm import tqdm
import time
import math

def evaluate(df, tool_col, desc_col):

    RPM = 5
    SLEEP = 60 / RPM
    # Collect all unique labels
    start_total = time.perf_counter()

    all_labels = TARGET_LIST
    print(all_labels)

    mlb = MultiLabelBinarizer(classes=all_labels)
    print(mlb.classes)

    # Predict single label per tool
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            pred = classify_tool(row[tool_col], row[desc_col])
            preds.append(pred)
            #time.sleep(SLEEP)


    df['predicted_label'] = preds
    df['predicted_label'] = df['predicted_label'].apply(parse_list)


    print(f"+++predicted label+++\n{df['predicted_label'][5]}")
    print(f"+++tool category+++\n {df['tool_category'][5]}")

    v = df.loc[1, 'predicted_label']
    print(type(v), v)

    z = df.loc[1, 'tool_category']
    print(type(z), z)
    # Binarize true and predicted labels
    y_true = mlb.fit_transform(df['tool_category'])
    y_pred = mlb.transform(df['predicted_label'])

    #print(f" y_pred = {y_pred}")
    #print(f" y_true = {y_true}")

    # Compute metrics

    subset_acc =  accuracy_score(y_true, y_pred)
    hamm_loss = hamming_loss(y_true, y_pred)

    micro_avg_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_avg_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_avg_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    macro_avg_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_avg_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_avg_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    class_report = classification_report(y_true, y_pred, target_names=all_labels, zero_division=0)

    end_total = time.perf_counter()
    total_time_sec = end_total - start_total

    metrics = {
        "subset_accuracy": subset_acc,
        "hamming_loss": hamm_loss,
        "micro_precision": micro_avg_precision,
        "micro_recall": micro_avg_recall,
        "micro_f1": micro_avg_f1,
        "macro_precision": macro_avg_precision,
        "macro_recall": macro_avg_recall,
        "macro_f1": macro_avg_f1,
        "n_samples": len(df),
        "total_time_sec": total_time_sec
    }

    save_eval(metrics, class_report, out_dir = "eval_logs", run_name="mistral_single_label")

data = pd.read_csv("gpt_final_dataset.csv", converters={"tool_category": ast.literal_eval})

tool_name = "tool"
tool_description = "tool_description"

#subset = data.head(5).copy()

evaluate(data, tool_name, tool_description)