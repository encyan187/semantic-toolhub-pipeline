from torch import nn
from transformers import BertModel
from new_config import PRETRAINED_MODEL_NAME


class SemanticBERT(nn.Module):
    def __init__(self):
        super(SemanticBERT, self).__init__()
        self.bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME , return_dict=True)
        hidden = self.bert_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 36)
        )

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask, token_type_ids)
        pooled = self._mean_pool(output.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return logits