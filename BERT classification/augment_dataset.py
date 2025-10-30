import random
import numpy as np
from new_augment_func import augment_minority_data
from skmultilearn.model_selection import iterative_train_test_split
from new_config import *
from new_utils import *

# Load and split dataset
def augment_dataset(dataset_path):

    data = pd.read_csv(dataset_path)

    data = data.drop(columns=['tool'])

    label_cols = [col for col in data.columns if col != TEXT_COLUMN]

    print(label_cols)

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    print("Category counts in train_df:")
    print(data[label_cols].sum().sort_values(ascending=False))

    data = augment_minority_data(data, TEXT_COLUMN)

    data.to_csv(AUGMENT_DATASET_PATH, index=False)

    return data

augment_dataset(DATASET_PATH)