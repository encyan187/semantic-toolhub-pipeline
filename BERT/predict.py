import torch
from config import *


def predict(text, model, tokenizer, max_len, threshold=0.3):
    encodings = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True,
        return_tensors="pt"
    )
    model.eval()
    with torch.no_grad():
        input_ids = encodings["input_ids"].to(DEVICE, dtype=torch.long)
        attention_mask = encodings["attention_mask"].to(DEVICE, dtype=torch.long)
        token_type_ids = encodings["token_type_ids"].to(DEVICE, dtype=torch.long)
        outputs = model(input_ids, attention_mask, token_type_ids)
        final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
        print(final_outputs)

    all_predictions = []

    for probs in final_outputs:
        labels = [
            TARGET_LIST[i]
            for i, prob in enumerate(probs)
            if prob >= threshold
        ]
        all_predictions.append(labels)

    return all_predictions