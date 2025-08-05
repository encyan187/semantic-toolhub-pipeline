import ast
from gpt_config import TARGET_LIST
from gpt_utils import parse_list
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

def evaluate(df, tool_col, desc_col):

    # Collect all unique labels
    all_labels = TARGET_LIST
    print(all_labels)

    mlb = MultiLabelBinarizer(classes=all_labels)
    print(mlb.classes)

    # Predict single label per tool
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        pred = classify_tool(row[tool_col], row[desc_col])
        preds.append(pred)
    df['predicted_label'] = preds
    df['predicted_label'] = df['predicted_label'].apply(parse_list)

    print(df['predicted_label'])
    print(df['tool_category'])

    v = df.loc[1, 'predicted_label']
    print(type(v), v)

    z = df.loc[1, 'tool_category']
    print(type(z), z)
    # Binarize true and predicted labels
    y_true = mlb.fit_transform(df['tool_category'])
    y_pred = mlb.transform(df['predicted_label'])

    print(y_pred)
    print(y_true)

    # Compute metrics
    print("Subset accuracy (exact match):", accuracy_score(y_true, y_pred))
    print("Hamming loss:", hamming_loss(y_true, y_pred))
    print()
    print("Micro-average precision:", precision_score(y_true, y_pred, average='micro', zero_division=0))
    print("Micro-average recall:   ", recall_score(y_true, y_pred, average='micro', zero_division=0))
    print("Micro-average F1:       ", f1_score(y_true, y_pred, average='micro', zero_division=0))
    print()
    print("Macro-average precision:", precision_score(y_true, y_pred, average='macro', zero_division=0))
    print("Macro-average recall:   ", recall_score(y_true, y_pred, average='macro', zero_division=0))
    print("Macro-average F1:       ", f1_score(y_true, y_pred, average='macro', zero_division=0))
    print()
    print("Classification report per category:")
    print(classification_report(y_true, y_pred, target_names=all_labels, zero_division=0))

data = pd.read_csv("gpt_final_dataset.csv", converters={"tool_category": ast.literal_eval})

tool_name = "tool"
tool_description = "tool_description"

subset = data.head(2).copy()

evaluate(subset, tool_name, tool_description)