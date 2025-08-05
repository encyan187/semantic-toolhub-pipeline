import pandas as pd
from torch.utils.data import DataLoader
from new_config import TOKENIZER, MAX_SEQ_LENGTH
from new_dataset import CustomDataset
import numpy as np

def load_data_loader(path, shuffle, batch_size):

    data = pd.read_csv(path)

    label_columns = [col for col in data.columns if col != "tool_description"]
    data[label_columns] = data[label_columns].astype(np.float32)

    dataloader = DataLoader(CustomDataset(data, tokenizer = TOKENIZER, max_seq_length= MAX_SEQ_LENGTH), batch_size=batch_size, shuffle=shuffle)

    return dataloader