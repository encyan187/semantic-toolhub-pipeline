import random
import numpy as np
from new_augment_func import augment_minority_data
from skmultilearn.model_selection import iterative_train_test_split
from new_config import *
from new_utils import *

# Load and split dataset
def split_dataset(dataset_path, train_size):

    data = pd.read_csv(dataset_path)

    data = data.drop(columns=['tool'])

    X = data[TEXT_COLUMN].values.reshape(-1, 1)

    label_cols = [col for col in data.columns if col != TEXT_COLUMN]

    print(label_cols)

    y = data[label_cols].values

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Perform stratified split
    X_train, y_train, X_val, y_val = iterative_train_test_split(X, y, test_size=1 - train_size)

    X_val = X_val.astype(str)

    print(X_val[:5])

    # Rebuild dataframes
    train_df = pd.DataFrame(X_train, columns=[TEXT_COLUMN])
    val_df = pd.DataFrame(X_val, columns=[TEXT_COLUMN])

    for i, col in enumerate(label_cols):
        train_df[col] = y_train[:, i]
        val_df[col] = y_val[:, i]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("Category counts in train_df:")
    print(train_df[label_cols].sum().sort_values(ascending=False))
    print("\nCategory counts in val_df:")
    print(val_df[label_cols].sum().sort_values(ascending=False))

    train_df = augment_minority_data(train_df, TEXT_COLUMN)

    train_df.to_csv(TRAIN_DATASET_PATH, index=False)
    val_df.to_csv(VAL_DATASET_PATH, index=False)

    return train_df, val_df

split_dataset(DATASET_PATH, TRAIN_SIZE)