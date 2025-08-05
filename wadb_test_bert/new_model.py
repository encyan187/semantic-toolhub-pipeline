from torch import nn
from transformers import BertModel

from new_config import PRETRAINED_MODEL_NAME


class SemanticBERT(nn.Module):
    def __init__(self):
        super(SemanticBERT, self).__init__()
        self.bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME , return_dict=True)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 36)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask, token_type_ids)
        cls_output = output.pooler_output
        logits = self.classifier(cls_output)
        return logits