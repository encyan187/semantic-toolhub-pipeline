import torch
from config import *
from transformers import BertTokenizer, BertForSequenceClassification

def load_model():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_LABELS = len(TARGET_LIST)

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #Load Checkpoint
    ckpt = torch.load(MODEL_DIR, map_location=DEVICE)
    state_dict = ckpt["state_dict"]

    # Recreate the architecture
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=NUM_LABELS
    )

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    return model, tokenizer