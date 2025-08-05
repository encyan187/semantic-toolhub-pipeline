import torch
from utils import compute_metrics
from config import DEVICE

def test(model, dataloader):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(DEVICE, dtype=torch.long)

            targets = batch["targets"].to(DEVICE, dtype=torch.float)
            outputs = model(input_ids, attention_mask, token_type_ids)
            outputs = torch.sigmoid(outputs)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    return print(metrics)