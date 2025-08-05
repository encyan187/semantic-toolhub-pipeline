from torch import nn
from transformers import BertModel

class SemanticBERT(nn.Module):
    def __init__(self):
        super(SemanticBERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased" , return_dict=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768,36)

    def forward(self , input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask, token_type_ids)
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output