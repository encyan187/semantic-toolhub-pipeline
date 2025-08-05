import pandas as pd
import numpy as np
from config import *

from augment_func import augment_minority_data

# Load and split dataset

data = pd.read_csv(DATASET_PATH)
data.drop(columns="tool", inplace=True, axis=1)

train_size = TRAIN_SIZE

np.random.seed(42)

train_df = data.sample(frac = train_size).reset_index(drop = True)
remaining_df = data.drop(train_df.index).reset_index(drop=True)

val_df = remaining_df.sample(frac=0.5).reset_index(drop=True)
test_df = remaining_df.drop(val_df.index).reset_index(drop=True)

# Augment training dataset

#train_df = augment_minority_data(train_df, TEXT_COLUMN)

print(len([col for col in train_df.columns if col != TEXT_COLUMN]))

train_df.to_csv("data/train.csv", index = False)
val_df.to_csv("data/val.csv", index = False)
test_df.to_csv("data/test.csv", index = False)



