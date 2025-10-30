import torch
from torch.utils.data import Dataset
from new_config import TARGET_LIST, TEXT_COLUMN

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.texts = self.df[TEXT_COLUMN]
        self.targets = self.df[TARGET_LIST].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }