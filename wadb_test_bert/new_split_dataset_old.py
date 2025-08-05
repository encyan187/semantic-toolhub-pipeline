import random
import numpy as np
from new_augment_func import augment_minority_data
from new_config import *
from new_utils import *

# Load and split dataset
def split_dataset(dataset_path, train_size):

    data = pd.read_csv(dataset_path)

    data.drop(columns="tool", inplace=True, axis=1)

    train_size = train_size

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    train_df = data.sample(frac = train_size).reset_index(drop = True)
    remaining_df = data.drop(train_df.index).reset_index(drop=True)

    val_df = remaining_df.sample(frac=0.5).reset_index(drop=True)
    test_df = remaining_df.drop(val_df.index).reset_index(drop=True)

    # Augment training dataset

    train_df = augment_minority_data(train_df, TEXT_COLUMN)

    train_df.to_csv(TRAIN_DATASET_PATH, index = False)
    val_df.to_csv(VAL_DATASET_PATH, index = False)
    test_df.to_csv(TEST_DATASET_PATH, index = False)

    return train_df, val_df, test_df



