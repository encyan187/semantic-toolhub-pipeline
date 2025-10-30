
import pandas as pd
import nlpaug.augmenter.word as naw
import ast
from collections.abc import Iterable

augmenter = naw.SynonymAug(aug_src='wordnet')

def parse_list(x):

    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
            if isinstance(obj, Iterable) and not isinstance(obj, (bytes, bytearray)):
                return list(obj)
            return [obj]
        except (ValueError, SyntaxError):
            return [x]
    if isinstance(x, Iterable) and not isinstance(x, (bytes, bytearray)):
        return list(x)
    return [x]

def augment_minority_data(df, text_column, augmenter=augmenter):
    if augmenter is None:
        augmenter = augmenter

    # Identify label columns (everything except text column)
    label_columns = [col for col in df.columns if col != text_column]

    # Count samples per label
    label_counts = df[label_columns].sum().sort_values()
    max_count = label_counts.max()

    augmented_rows = []

    for label in label_columns:
        count = label_counts[label]
        extra_needed = max_count - count

        if extra_needed <= 0:
            continue  # Already balanced

        # Select rows where this label is 1
        candidates = df[df[label] == 1]

        if len(candidates) == 0:
            continue  # skip if no candidates available

        sampled = candidates.sample(n=extra_needed, replace=True, random_state=42)

        for _, row in sampled.iterrows():
            aug_text = augmenter.augment(row[text_column])

            if not aug_text or (isinstance(aug_text, list) and len(aug_text) == 0):
                continue

            if isinstance(aug_text, list):
                aug_text = aug_text[0]

            if aug_text.strip() == "" or aug_text.strip() == "[]" or aug_text.strip() == "  []":
                continue

            new_row = row.copy()
            new_row[text_column] = aug_text
            augmented_rows.append(new_row)

    # Combine original + augmented
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        balanced_df = pd.concat([df, aug_df], ignore_index=True)
    else:
        balanced_df = df.copy()

    balanced_df.dropna(subset=[text_column], inplace=True)
    balanced_df = balanced_df[~balanced_df[text_column].isin(["", "[]"])]
    balanced_df = balanced_df[balanced_df[text_column].apply(lambda x: isinstance(x, str) and x.strip() != "[]")]

    return balanced_df


def augment_whole_dataset(df,text_column,label_column,n_augment = 1,augmenter=augmenter):

    if augmenter is None:
        augmenter = augmenter

    df_copy = df.copy()
    df_copy[label_column] = df_copy[label_column].apply(parse_list)

    augmented_rows = []

    for _, row in df_copy.iterrows():
        for _ in range(n_augment):
            aug_text = augmenter.augment(row[text_column])
            augmented_rows.append({
                text_column: aug_text,
                label_column: row[label_column]
            })

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        full_augmented = pd.concat([df_copy, aug_df], ignore_index=True)
    else:
        full_augmented = df_copy

    return full_augmented
